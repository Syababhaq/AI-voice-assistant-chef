import json
import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# CONFIGURATION
JSON_PATH = "docs/recipes.json"
DB_PATH = "vector_db"

def create_vector_db():
    print(f"Loading {JSON_PATH}...")
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    # 1. Load JSON Data
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    documents = []
    
    # 2. Convert each Recipe into a "Document" for the AI
    # We combine all fields into a single text block so the AI reads it as one unit.
    for recipe in data['recipes']:
        text_content = f"""
        Recipe Name: {recipe['name']} (Aliases: {', '.join(recipe['aliases'])})
        Category: {recipe['category']}
        Difficulty: {recipe['difficulty']}
        Prep Time: {recipe['prep_time_minutes']} mins | Cook Time: {recipe['cook_time_minutes']} mins
        
        Ingredients:
        {', '.join(recipe['ingredients'])}
        
        Steps:
        {' '.join(recipe['steps'])}
        """

        # Create a LangChain Document object
        doc = Document(page_content=text_content, metadata={"name": recipe['name']})
        documents.append(doc)

    print(f"Processed {len(documents)} recipes.")

    # 3. Create Embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Create and Save the Database
    # Note: We skipped the 'Splitter' because each recipe is small enough to fit in one chunk!
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH) # Clear old database to prevent duplicates
        
    db = Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
    print(f"Success! Recipe Database saved to '{DB_PATH}'.")

if __name__ == "__main__":
    create_vector_db()
