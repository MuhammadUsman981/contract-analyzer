# sanity_check.py – verify core libs in contract-analyzer env
import spacy
import faiss
import wandb
import fastapi
import uvicorn
import numpy as np

print("✅ All core libraries imported successfully")

# FAISS test
dim = 4
index = faiss.IndexFlatL2(dim)
vectors = np.array([[1,2,3,4], [2,3,4,5]], dtype='float32')
index.add(vectors)
D, I = index.search(np.array([[1,2,3,4]], dtype='float32'), k=2)

print("FAISS search distances:", D)
print("FAISS search indices:", I)

# spaCy test
nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a test sentence.")
print("spaCy tokens:", [t.text for t in doc])