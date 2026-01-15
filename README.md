# 🍳 AI AI Chef: Your Personal AI Voice-Powered Sous-Chef

**A Hybrid Neuro-Symbolic Cooking Assistant** designed to solve the "messy hands" problem in the kitchen. By combining deterministic logic with generative AI, this system provides a hands-free, hallucination-free cooking experience that never forgets which step you are on.

---

## 📖 Introduction
Cooking is inherently hands-on and often messy, making traditional cookbooks or touchscreens unsanitary and impractical. While general-purpose voice assistants (like Siri or Alexa) exist, they are ill-suited for the kitchen due to:
* **Hallucination Risk:** Inventing dangerous or non-existent ingredients.
* **State Amnesia:** Forgetting the current recipe step during long pauses.
* **Noise Sensitivity:** Misunderstanding commands in a loud kitchen.

**AI AI Chef** solves these issues by decoupling **deterministic control logic** (for accuracy) from **probabilistic generative AI** (for interaction), ensuring reliable, safe, and engaging cooking support.

---

## 🚀 Key Features
* **🖐️ 100% Hands-Free Control:** Wake word detection and voice navigation mean you never touch a screen with dough-covered hands.
* **🧠 Zero Hallucination (RAG):** Uses **Retrieval-Augmented Generation** to fetch facts from a trusted local dataset (VectorDB). The AI cannot invent recipes—it only reads what is verified.
* **💾 Context Awareness (FSM):** A **Finite State Machine** manages memory, ensuring the bot never loses track of your progress (e.g., it remembers you are on Step 4 of 10).
* **🔒 Privacy-First & Offline:** Runs locally using **Vosk** (STT) and **Ollama** (LLM), keeping your data private and reducing latency.
* **👨‍🍳 Dynamic Persona:** Retrieves dry factual steps and "chefs them up" using **Llama 3.2** to sound like a professional sous-chef.

---

## 🛠️ System Architecture

The system follows a strict processing pipeline to ensure accuracy:

1.  **Input Speech:** User voice commands are captured.
2.  **STT (Perception):** **Vosk-small** converts audio to text locally (Offline & Fast).
3.  **The Brain (Logic Core):**
    * **Python Logic:** Acts as the traffic controller.
    * **RAG Module:** Uses **LangChain**, **ChromaDB**, and **SBERT (all-MiniLM-L6-v2)** to perform semantic vector searches on the recipe dataset.
    * **FSM:** Tracks the "Awake/Asleep" state and "Current Recipe Step".
4.  **LLM (Generation):** **Llama 3.2** (running via Ollama) styles the text response.
5.  **TTS (Output):** **pyttsx3** converts the final text back to speech.

Architecture Flow
<img width="1587" height="2245" alt="AI AI Chef" src="https://github.com/user-attachments/assets/ce5981cb-1805-4b77-911c-0214a2a0c806" />


---

## 💻 Tech Stack
| Component | Technology Used |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Speech-to-Text** | Vosk (Model: `vosk-model-small-en-us-0.15`) |
| **LLM Inference** | Ollama (Model: `llama3.2`) |
| **Vector DB** | ChromaDB |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Orchestration** | LangChain |
| **Text-to-Speech** | pyttsx3 |

---

## 📥 Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Syababhaq/Al-voice-assistant-chef](https://github.com/Syababhaq/Al-voice-assistant-chef)
    cd Al-voice-assistant-chef
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Ollama (Required for LLM)**
    * Download and install [Ollama](https://ollama.com).
    * Pull the Llama 3.2 model:
        ```bash
        ollama pull llama3.2
        ```

4.  **Download Vosk Model**
    * Download `vosk-model-small-en-us-0.15` from the [Vosk Models page](https://alphacephei.com/vosk/models).
    * Extract it to a `models/` folder in the project directory.

---

## 🗣️ Usage Example

**Start the system:**
```bash
python main.py
```

## 🗣️ Interaction Flow

1.  **Wake Up:** "Hey Chef!"
2.  **Search:** "I want to cook spaghetti."
    * *System searches RAG... Match found: 'Spaghetti Bolognese' (Confidence: 0.1151)*
    * *Chef:* "I found Simple Spaghetti Bolognese. Do you want to cook this?"
3.  **Cook:** "Yes."
4.  **Navigate:** "Next step" or "Repeat that".
5.  **Rejection Logic:**
    * *User:* "Cook egg."
    * *System:* Confidence too low (Score > Threshold).
    * *Chef:* "I heard 'egg', but I'm not sure. Please try again."

---

## 👥 Team Members
* **Ikhwan (2218845)**
* **Syabab (2211117)**
* **Zamir (2212985)**

---

## 🔗 References
* **Vosk API:** [alphacephei.com/vosk/](https://alphacephei.com/vosk/)
* **RAG Concepts:** [Google Cloud RAG Use Cases](https://cloud.google.com/use-cases/retrieval-augmented-generation)
