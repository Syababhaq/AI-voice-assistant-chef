# run this code to download local SBERT all-MiniLM-L6-v2

from langchain_community.embeddings import HuggingFaceEmbeddings

# This will download the model files to a folder named "local_embeddings"
print("Downloading model... please wait.")
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
model.client.save("local_embeddings")
print("Download complete! You can now go offline.")
