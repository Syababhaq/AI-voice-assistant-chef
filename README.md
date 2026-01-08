# AI-voice-assistant-chef

## 1. Input Layer
-	Vosk (Small Model): Listens to your microphone.
-	Vocabulary Restriction: It filters sounds against your CHEF_VOCAB list. If you say "Automobile," it ignores it. If you say "Chicken," it captures it.
-	Wake Word: In "Sleep Mode," it only reacts to "Hey Chef." In "Awake Mode," it processes everything.

## 2. The Logic Core (process_request)
Once text is captured, Python makes a decision based on Priority States:

State A: Pending Confirmation (Highest Priority)

-	Condition: Is there a recipe waiting in the pending variable?
-	Action: It listens only for "Yes" or "No".
-	Yes: Locks the recipe into current_active_recipe and resets the step counter.
-	No: Discards the data.
-	Bypass LLM: True (Keeps the question "Do you want to cook this?" simple).

State B: Semantic Search (RAG)

-	Condition: You said "Cook", "Make", or "Recipe" (and didn't say "Step").
-	Cleaning: It strips words like "I", "want", "to" (to fix the "Fred" vs "Fried" bug).
-	Search: Queries ChromaDB with a Score Threshold (1.0).
-	Bad Score: "I don't have that recipe."
-	Good Score: Finds the recipe name, loads it from JSON, and moves to State A.

State C: Navigation (Step Controller)

-	Condition: A recipe is active.
-	Logic: It uses pure Python Math (index += 1 or index -= 1).
-	Data: It extracts the text from your JSON, handling both Strings and Dictionaries.
-	Bypass LLM: False (It sends the step text to Llama to be "Chef-ified").

## 3. The Personality Layer (Llama 3.2)
-	Decision: The system checks the bypass_llm flag.
-	If Bypass (True): The system speaks raw text (e.g., "I found Nasi Goreng. Do you want to cook this?"). This prevents hallucination during critical confirmations.
-	If Standard (False): The text is sent to Llama 3.2 with the prompt: "Rewrite this to sound like a professional Chef."
-	Input: "Step 1: Heat oil."
-	Output: "Right then! Step one, get that oil hot in the pan!"

## 4. Output Layer (The Mouth)
-	Cleaner: The speak() function strips *, #, and - symbols so the voice doesn't read them out loud.
-	TTS: pyttsx3 generates the audio file.
-	Playback: pygame plays the audio (blocking input until finished, unless you press 's' to stop).
