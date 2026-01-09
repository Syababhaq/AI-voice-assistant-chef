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

