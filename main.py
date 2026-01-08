import os
import json
import time
import uuid
import pyaudio
import pyttsx3
import ollama
import pygame
import keyboard
import re
from vosk import Model, KaldiRecognizer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION ---
MODEL_PATH = r"models\vosk-model-small-en-us-0.15" 
DB_PATH = "vector_db"
JSON_PATH = "docs/recipes.json"
LLM_MODEL = "llama3.2"

# --- VOCABULARY LIST ---
CHEF_VOCAB = str([
    # System
    "hey chef", "chef", "stop listening", "goodbye", "exit", "sleep", "wake up",
    "stop", "cancel", "standing by", "system", "[unk]",
    "thank you", "thanks",
    # Navigation
    "yes", "no", "next", "repeat", "continue", "back", "previous", "step", "ready", "okay", "one", "two", "three", "four", "five",
    # Triggers
    "cook", "make", "fry", "fried", "boil", "recipe", "ingredients", "how", "want", "i", "have", "need",
    # Ingredients (From your JSON)
    "rice", "egg", "eggs", "garlic", "oil", "soy", "sauce", "salt", "noodles", "bread",
    "butter", "chicken", "water", "pepper", "mayonnaise", "milk", "sugar", "tea", "bag",
    "toast", "soup", "sandwich", "omelette", "spaghetti", "cheese", "bolognese", "mac"
]).replace("'", '"')

# --- GLOBAL STATE ---
current_active_recipe = None 
pending_recipe = None       # <--- NEW: Stores the recipe waiting for "Yes/No"
current_step_index = -1 

# 1. SETUP DATABASE
print("Loading cookbook...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

with open(JSON_PATH, 'r') as f:
    full_cookbook = json.load(f)

def get_recipe_by_name(name):
    for r in full_cookbook['recipes']:
        if r['name'] == name: return r
    return None

# 2. SETUP MODEL & AUDIO
if not os.path.exists(MODEL_PATH): exit(1)
vosk_model = Model(MODEL_PATH)
pygame.mixer.init()

def speak(text):
    # --- 1. CLEANING STEP ---
    # Remove Markdown symbols that sound bad
    text = text.replace("*", "")       # Remove asterisks
    text = text.replace("#", "")       # Remove headers
    text = text.replace("`", "")       # Remove code blocks
    text = text.replace("- ", "")      # Remove dash bullet points
    text = text.strip()                # Remove extra spaces
    
    print(f"Chef: {text}")             # Print clean text
    
    # --- 2. AUDIO GENERATION ---
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

# 3. DETERMINISTIC LOGIC ENGINE (WITH CONFIRMATION)
def process_request(user_input):
    global current_active_recipe, current_step_index, pending_recipe
    
    user_text = user_input.lower()
    response_text = ""
    bypass_llm = False # Flag to skip LLM rewriting (for crisp questions)

    # --- HELPER: Clean Text Extraction from JSON ---
    def get_step_content(step_data):
        if isinstance(step_data, str): return step_data
        elif isinstance(step_data, dict): return step_data.get('instruction', step_data.get('step', str(step_data)))
        return str(step_data)

    # --- STATE A: PENDING CONFIRMATION ---
    if pending_recipe:
        # User says YES
        if any(w in user_text for w in ["yes", "sure", "okay", "ready", "correct", "cook"]):
            current_active_recipe = pending_recipe 
            pending_recipe = None                  
            current_step_index = -1                
            
            ing_list = ", ".join(current_active_recipe['ingredients'])
            response_text = f"Excellent. Ingredients are: {ing_list}. Say 'Ready' for Step 1."
            # bypass_llm = False (Let the Chef personality read the ingredients)
        
        # User says NO
        elif any(w in user_text for w in ["no", "stop", "cancel", "wrong"]):
            pending_recipe = None
            response_text = "Dish discarded. What else do you want to cook?"
            bypass_llm = True # Quick response
        
        else:
            response_text = f"I need a Yes or No. Do you want to cook {pending_recipe['name']}?"
            bypass_llm = True

        return response_text, bypass_llm

    # --- STATE B: NEW RECIPE SEARCH ---
    new_triggers = ["cook", "make", "recipe", "want", "ingredients"]
    
    # Check if any trigger word is present, but ensure we aren't mid-recipe (checking for 'step')
    if any(t in user_text for t in new_triggers) and "step" not in user_text:
        
        # --- BUG FIX: REMOVE WHOLE WORDS ONLY ---
        stop_words = ["cook", "make", "recipe", "ingredients", "want", "i", "to", "please", "can", "you", "would", "like", "a", "the"]
        
        # Split sentence into a list of words: "cook fried rice" -> ["cook", "fried", "rice"]
        words = user_text.split()
        
        # Keep only words that are NOT in the stop_words list
        filtered_words = [w for w in words if w not in stop_words]
        
        # Join them back together: ["fried", "rice"] -> "fried rice"
        cleaned_text = " ".join(filtered_words).strip()
            
        if len(cleaned_text) < 3:
            return "What do you want to cook?", True
            
        print(f"--> [Searching RAG for '{cleaned_text}']")
        
        # Search with Score (keep this, it's good safety!)
        docs = db.similarity_search_with_score(cleaned_text, k=1)
        
        if docs:
            doc, score = docs[0]
            recipe_name = doc.metadata.get('name')
            print(f"--> [Match Found: '{recipe_name}' | Score: {score:.4f}]")
            
            # Threshold Check (1.0 is a decent limit)
            if score > 1.0: 
                print("--> [Rejected: Score too high]")
                return f"I heard '{cleaned_text}', but I don't have a recipe for that.", True

            found_recipe = get_recipe_by_name(recipe_name)
            
            if found_recipe:
                pending_recipe = found_recipe 
                response_text = f"I found {recipe_name}. Do you want to cook this?"
                bypass_llm = True 
            else:
                response_text = f"Found {recipe_name}, but data is missing."
                bypass_llm = True
        else:
             response_text = "I don't have that recipe."
             bypass_llm = True
             
        return response_text, bypass_llm

    # --- STATE C: NAVIGATION ---
    elif current_active_recipe:
        total_steps = len(current_active_recipe['steps'])
        
        # NEXT
        if any(w in user_text for w in ["next", "yes", "ready", "step", "okay", "continue"]):
            if current_step_index < total_steps - 1:
                current_step_index += 1
                raw_step = current_active_recipe['steps'][current_step_index]
                step_text = get_step_content(raw_step)
                response_text = f"Step {current_step_index + 1}: {step_text}"
            else:
                response_text = "That was the last step. Enjoy your meal!"

        # PREVIOUS
        elif any(w in user_text for w in ["back", "previous"]):
            if current_step_index > 0:
                current_step_index -= 1
                raw_step = current_active_recipe['steps'][current_step_index]
                step_text = get_step_content(raw_step)
                response_text = f"Step {current_step_index + 1}: {step_text}"
            else:
                response_text = "We are at the beginning."

        # REPEAT
        elif "repeat" in user_text:
            if current_step_index == -1:
                 ing_list = ", ".join(current_active_recipe['ingredients'])
                 response_text = f"Ingredients: {ing_list}"
            else:
                raw_step = current_active_recipe['steps'][current_step_index]
                step_text = get_step_content(raw_step)
                response_text = f"Repeating Step {current_step_index + 1}: {step_text}"

        return response_text, False # Let Chef personality handle steps

    return "Sorry I don't have that information. Say 'Cook [Dish Name]'.", True

# --- MAIN LOOP ---
if __name__ == "__main__":
    is_awake = False
    print(f"\n[System Online] Waiting for 'Hey Chef'...")
    speak("System Online. Waiting for Order.")
    
    try:
        while True:
            if not is_awake:
                user_text = listen('["hey chef", "chef", "goodbye", "[unk]"]')
                if user_text:
                    if "chef" in user_text:
                        speak("Yes Chef?")
                        is_awake = True
                    elif "goodbye" in user_text:
                        speak("Kitchen closed.")
                        break
            else:
                user_text = listen()
                if user_text:
                    print(f"Heard: {user_text}")
                    if "goodbye" in user_text: 
                        speak("Goodbye Chef.")
                        break
                    if "sleep" in user_text:
                        is_awake = False
                        speak("Standing by.")
                        continue
                        
                    # 1. Get Logic Response
                    response_text, bypass_llm = process_request(user_text)
                    
                    if response_text:
                        # 2. Decide: Speak Directly OR Use LLM?
                        if bypass_llm:
                            # Direct Speak (Fast, Accurate, No Hallucination)
                            speak(response_text)
                        else:
                            # Personality Rewrite
                            prompt = f"""
                            Rewrite this instruction to sound like a professional Chef. 
                            Keep it short and clear. Do not add extra steps.
                            
                            Instruction: "{response_text}"
                            """
                            try:
                                llm_response = ollama.chat(model=LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
                                speak(llm_response['message']['content'])
                            except:
                                speak(response_text) # Fallback

    except KeyboardInterrupt:
        print("\nStopping...")
