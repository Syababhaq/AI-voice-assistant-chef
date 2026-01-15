import os
import json
import time
import uuid
import pyaudio
import pyttsx3
import ollama
import pygame
import keyboard
import threading
import tkinter as tk
from tkinter import scrolledtext, font
from vosk import Model, KaldiRecognizer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION ---
MODEL_PATH = r"models\vosk-model-small-en-us-0.15" 
DB_PATH = "vector_db"
JSON_PATH = "docs/recipes.json"
LLM_MODEL = "llama3.2"

# --- GUI GLOBAL REFERENCE ---
gui_app = None  # Logic will talk to this variable

# --- VOCABULARY LIST ---
CHEF_VOCAB = str([
    "hey chef", "chef", "stop listening", "goodbye", "exit", "sleep", "wake up",
    "stop", "cancel", "standing by", "system", "[unk]",
    "thank you", "thanks", "finish", "done", "complete",
    "yes", "no", "next", "repeat", "continue", "back", "previous", "step", "ready", "okay", "one", "two", "three", "four", "five",
    "cook", "make", "fry", "fried", "boil", "recipe", "ingredients", "how", "want", "i", "have", "need",
    "rice", "egg", "eggs", "garlic", "oil", "soy", "sauce", "salt", "noodles", "bread",
    "butter", "chicken", "water", "pepper", "mayonnaise", "milk", "sugar", "tea", "bag",
    "toast", "soup", "sandwich", "omelette", "spaghetti", "cheese", "bolognese", "mac"
]).replace("'", '"')

# --- GLOBAL STATE ---
current_active_recipe = None 
pending_recipe = None       
current_step_index = -1 

# --- SETUP DATABASE ---
print("Loading cookbook...")
embeddings = HuggingFaceEmbeddings(model_name="./local_embeddings")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

with open(JSON_PATH, 'r') as f:
    full_cookbook = json.load(f)

def get_recipe_by_name(name):
    for r in full_cookbook['recipes']:
        if r['name'] == name: return r
    return None

# --- SETUP AUDIO ---
if not os.path.exists(MODEL_PATH): exit(1)
vosk_model = Model(MODEL_PATH)
pygame.mixer.init()

# --- GUI LOGGING HELPER ---
def log(text, source="System"):
    print(f"[{source}] {text}")
    if gui_app:
        gui_app.update_terminal(f"[{source}] {text}")

# --- SPEAK FUNCTION ---
def speak(text):
    text = text.replace("*", "").replace("#", "").replace("`", "").replace("- ", "").strip()
    log(text, "Chef") # Log to GUI
    
    filename = f"response_{uuid.uuid4().hex}.wav"
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    engine.save_to_file(text, filename)
    engine.runAndWait()
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if keyboard.is_pressed('s'):
                pygame.mixer.music.stop()
                break
            time.sleep(0.1)
    except: pass
    finally:
        pygame.mixer.music.unload()
        try: os.remove(filename)
        except: pass

# --- LISTEN FUNCTION ---
def listen(valid_words=None):
    vocab_to_use = valid_words if valid_words else CHEF_VOCAB
    rec = KaldiRecognizer(vosk_model, 16000, vocab_to_use)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()
    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result['text']
                if text:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return text
    except: return ""

# --- LOGIC ENGINE ---
def process_request(user_input):
    global current_active_recipe, current_step_index, pending_recipe
    
    user_text = user_input.lower()
    response_text = ""
    bypass_llm = False

    def get_step_content(step_data):
        if isinstance(step_data, str): return step_data
        elif isinstance(step_data, dict): return step_data.get('instruction', str(step_data))
        return str(step_data)

    # STATE A: PENDING CONFIRMATION
    if pending_recipe:
        if any(w in user_text for w in ["yes", "sure", "okay", "ready", "cook"]):
            current_active_recipe = pending_recipe 
            pending_recipe = None                  
            current_step_index = -1                
            
            # --- GUI UPDATE: Show Recipe ---
            if gui_app: gui_app.show_recipe_view(current_active_recipe)
            
            ing_list = ", ".join(current_active_recipe['ingredients'])
            return f"Excellent. Ingredients are: {ing_list}. Say 'Ready' for Step 1.", False
        
        elif any(w in user_text for w in ["no", "stop", "cancel"]):
            pending_recipe = None
            if gui_app: gui_app.show_menu_view() # Back to Menu
            return "Dish discarded. What else?", True
        
        else:
            return f"I need a Yes or No. Cook {pending_recipe['name']}?", True

    # STATE B: SEARCH
    new_triggers = ["cook", "make", "recipe", "want", "ingredients"]
    if any(t in user_text for t in new_triggers) and "step" not in user_text:
        stop_words = ["cook", "make", "recipe", "ingredients", "want", "i", "to", "please", "can", "you", "would", "like", "a", "the"]
        words = user_text.split()
        filtered_words = [w for w in words if w not in stop_words]
        cleaned_text = " ".join(filtered_words).strip()
            
        if len(cleaned_text) < 3: return "What do you want to cook?", True
        
        log(f"Searching RAG for '{cleaned_text}'...", "System")
        docs = db.similarity_search_with_score(cleaned_text, k=1)
        
        if docs:
            doc, score = docs[0]
            recipe_name = doc.metadata.get('name')

            # --- CONVERT TO SIMILARITY % ---
            similarity = 1.0 - score
            if similarity < 0: similarity = 0 # Clamp to 0 if distance is huge

            log(f"Match: '{recipe_name}' | Confidence: {similarity:.4f}", "System")
            
            if score > 1.0: return f"I heard '{cleaned_text}', but I'm not sure. Please try again.", True

            found_recipe = get_recipe_by_name(recipe_name)
            if found_recipe:
                pending_recipe = found_recipe 
                return f"I found {recipe_name}. Do you want to cook this?", True
            else: return f"Found {recipe_name}, but data is missing.", True
        else: return "I don't have that recipe.", True

    # STATE C: NAVIGATION
    elif current_active_recipe:
        total_steps = len(current_active_recipe['steps'])
        
        # --- MODIFIED NEXT / FINISH LOGIC ---
        if any(w in user_text for w in ["next", "yes", "ready", "step", "okay", "continue", "finish", "done"]):
            if current_step_index < total_steps - 1:
                # Normal Next Step
                current_step_index += 1
                raw_step = current_active_recipe['steps'][current_step_index]
                step_text = get_step_content(raw_step)
                
                if gui_app: gui_app.highlight_step(current_step_index)
                return f"Step {current_step_index + 1}: {step_text}", False
            else:
                # RECIPE FINISHED - RESET EVERYTHING
                current_active_recipe = None
                current_step_index = -1
                if gui_app: gui_app.show_menu_view() # <--- GO BACK TO MENU
                return "Recipe finished. Returning to menu. What would you like to cook next?", False

        elif any(w in user_text for w in ["back", "previous"]):
            if current_step_index > 0:
                current_step_index -= 1
                raw_step = current_active_recipe['steps'][current_step_index]
                step_text = get_step_content(raw_step)
                if gui_app: gui_app.highlight_step(current_step_index)
                return f"Step {current_step_index + 1}: {step_text}", False
            else: return "We are at the start.", False

        elif "repeat" in user_text:
            if current_step_index == -1:
                 ing_list = ", ".join(current_active_recipe['ingredients'])
                 return f"Ingredients: {ing_list}", False
            else:
                raw_step = current_active_recipe['steps'][current_step_index]
                step_text = get_step_content(raw_step)
                return f"Repeating Step {current_step_index + 1}: {step_text}", False

    return "Sorry I don't have that info.", True

# --- VOICE THREAD FUNCTION ---
def run_voice_assistant():
    is_awake = False
    log("Waiting for 'Hey Chef'...", "System")
    speak("System Online.")
    
    try:
        while True:
            if not is_awake:
                user_text = listen('["hey chef", "chef", "goodbye", "[unk]"]')
                if user_text:
                    if "chef" in user_text:
                        speak("Yes Chef?")
                        is_awake = True
                        if gui_app: gui_app.set_status("LISTENING", "green")
                    elif "goodbye" in user_text:
                        speak("Kitchen closed.")
                        os._exit(0)
            else:
                user_text = listen()
                if user_text:
                    log(f"Heard: {user_text}", "User")
                    
                    if "goodbye" in user_text: 
                        speak("Goodbye Chef.")
                        os._exit(0)
                    if "sleep" in user_text:
                        is_awake = False
                        speak("Standing by.")
                        if gui_app: gui_app.set_status("SLEEPING", "orange")
                        continue
                        
                    response_text, bypass_llm = process_request(user_text)
                    
                    if response_text:
                        if bypass_llm:
                            speak(response_text)
                        else:
                            prompt = f"Rewrite this instruction as a professional Chef. Keep short. Instruction: '{response_text}'"
                            try:
                                llm_response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
                                speak(llm_response['message']['content'])
                            except:
                                speak(response_text)
    except Exception as e:
        log(f"Error: {e}", "Error")

# --- GUI CLASS (DISPLAY ONLY) ---
class DisplayGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Sous-Chef Display")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")

        # 1. HEADER (Status)
        self.header_frame = tk.Frame(root, bg="#34495e", height=50)
        self.header_frame.pack(fill="x")
        self.status_lbl = tk.Label(self.header_frame, text="SYSTEM STATUS: SLEEPING", font=("Arial", 14, "bold"), fg="orange", bg="#34495e")
        self.status_lbl.pack(pady=10)

        # 2. MAIN CONTENT (Cookbook / Recipe)
        self.content_frame = tk.Frame(root, bg="#ecf0f1")
        self.content_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # 3. TERMINAL (Logs)
        self.term_label = tk.Label(root, text="Voice Logs", bg="#2c3e50", fg="white", anchor="w")
        self.term_label.pack(fill="x", padx=10)
        self.terminal = scrolledtext.ScrolledText(root, height=8, bg="black", fg="#00ff00", font=("Consolas", 10))
        self.terminal.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        
        self.step_labels = [] # To track step widgets for highlighting
        self.show_menu_view() # Start at menu

    def set_status(self, text, color):
        self.status_lbl.config(text=f"SYSTEM STATUS: {text}", fg=color)

    def update_terminal(self, text):
        self.terminal.insert(tk.END, text + "\n")
        self.terminal.see(tk.END)

    def show_menu_view(self):
        # Clear frame
        for widget in self.content_frame.winfo_children(): widget.destroy()
        
        lbl = tk.Label(self.content_frame, text="AVAILABLE RECIPES", font=("Arial", 24, "bold"), bg="#ecf0f1", fg="#2c3e50")
        lbl.pack(pady=20)
        
        # Simple List of Recipes
        for recipe in full_cookbook['recipes']:
            card = tk.Label(self.content_frame, text=f"🍳 {recipe['name']}", font=("Arial", 16), bg="white", width=40, pady=10, relief="raised")
            card.pack(pady=5)
        
        tk.Label(self.content_frame, text="(Say 'Hey Chef to wake up, 'sleep' to standby. Say 'Cook/Recipe [Recipe Name]' to start)", font=("Arial", 12, "italic"), bg="#ecf0f1").pack(pady=20)

    def show_recipe_view(self, recipe):
        # Clear frame
        for widget in self.content_frame.winfo_children(): widget.destroy()
        self.step_labels = []

        # Title
        tk.Label(self.content_frame, text=recipe['name'], font=("Arial", 22, "bold"), bg="#ecf0f1").pack(pady=10)

        # Split: Left (Ingredients) | Right (Steps)
        split_frame = tk.Frame(self.content_frame, bg="#ecf0f1")
        split_frame.pack(fill="both", expand=True)

        # Ingredients
        left_frame = tk.Frame(split_frame, bg="white", relief="sunken", bd=1, width=250)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)
        tk.Label(left_frame, text="Ingredients", font=("Arial", 14, "bold"), bg="white").pack(pady=5)
        for ing in recipe['ingredients']:
            tk.Label(left_frame, text=f"• {ing}", bg="white", anchor="w", font=("Arial", 11)).pack(fill="x", padx=5)

        # Steps
        right_frame = tk.Frame(split_frame, bg="#ecf0f1")
        right_frame.pack(side="right", fill="both", expand=True, padx=10)
        
        for idx, step in enumerate(recipe['steps']):
            # Clean step text if it's a dict
            step_txt = step['instruction'] if isinstance(step, dict) else str(step)
            lbl = tk.Label(right_frame, text=f"{idx+1}. {step_txt}", font=("Arial", 12), bg="#ecf0f1", anchor="w", justify="left", wraplength=450, padx=10, pady=5)
            lbl.pack(fill="x", pady=2)
            self.step_labels.append(lbl)

        tk.Label(self.content_frame, text="(Say 'next step' to contnue, or 'repeat' if want to repeat step.)", font=("Arial", 12, "italic"), bg="#ecf0f1").pack(pady=20)

    def highlight_step(self, index):
        # Reset all to default
        for lbl in self.step_labels: 
            lbl.configure(bg="#ecf0f1", fg="black", font=("Arial", 12))
        
        # Highlight current
        if 0 <= index < len(self.step_labels):
            self.step_labels[index].configure(bg="#3498db", fg="white", font=("Arial", 14, "bold"))

# --- STARTUP ---
if __name__ == "__main__":
    # Create GUI Root
    root = tk.Tk()
    gui_app = DisplayGUI(root) # Assign to global variable
    
    # Start Voice Logic in Background Thread
    t = threading.Thread(target=run_voice_assistant, daemon=True)
    t.start()
    
    # Start GUI Loop (Main Thread)
    root.mainloop()
