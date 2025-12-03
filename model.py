import os
import re
import ollama
import faiss
from pypdf import PdfReader
import numpy as np
import streamlit as st

#CONFIG - Data file and models
DATA_FILE = 'data/info.txt'
EMBED_MODEL = "all-minilm"      
GEN_MODEL = "llama3.2:1b"

#Helper Functions

#1. Reading the Info file
def read_file(path):
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    else:
        with open(path, "r", encoding="utf-8",errors="ignore") as f:
            text = f.read()
            # Remove the filename (or occurrences of the basename) from the text
            # so the LLM won't see and repeat the file name like "info.txt".
            try:
                basename = os.path.basename(path)
                if basename:
                    text = text.replace(basename, "")
            except Exception:
                pass
            # Clean up any excessive leading/trailing whitespace introduced
            return text.strip()
        
#2. Splitting the document into chunks
def chunk_text(text, size=800, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = min(start+size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

#3. Converts Chunks into Vectors (Embeddings)
def embed_texts(texts, model=EMBED_MODEL):
    embs = []
    for t in texts:
        resp = ollama.embeddings(model=model, prompt=t)
        embs.append(np.array(resp["embedding"], dtype=np.float32))
    return np.vstack(embs)

#4. Takes the embeddings and builds an index for quick retrieval
def build_faiss(embs):
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)   # cosine similarity after normalization
    faiss.normalize_L2(embs)
    index.add(embs)
    return index

# Takes the User query and retrieves the relevant embedding from the Index
def retrieve(query, index, chunks, k=6):
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=query)
    q_emb = np.array(resp["embedding"], dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    return [(chunks[i], float(D[0][j])) for j,i in enumerate(I[0])]

#
def generate_answer(query, retrieved):
    context = "\n\n".join([c[:1000] for c, _ in retrieved])

    system = (
        "You are a helpful assistant that speaks as Shreya Sheta (a Machine Learning student and practitioner). "
        "Answer in a direct, clear, and actionable way. Use only the information provided in the retrieved context or the user's explicit profile. "
        "If the answer is not present in the provided information, say 'I don't have that information' or 'I don't know' rather than inventing facts. "
        "Keep answers concise and practical. When relevant, include short actionable steps or commands. "
        "Never mention filenames (for example 'info.txt'), the source file name, or phrases like 'the document says' or 'based on context'. "
        "If you would otherwise reference a source name, instead answer directly using the provided information or state you don't know."
    )


    user_msg = f"{context}\n\nQuestion: {query}"

    resp = ollama.chat(model=GEN_MODEL, messages=[ 
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg}
    ])

    # Post-process the model output to remove accidental filename references
    ans = resp.message.content
    try:
        # Remove or replace explicit mentions of the data filename
        basename = os.path.basename(DATA_FILE)
        if basename and re.search(re.escape(basename), ans, flags=re.IGNORECASE):
            ans = re.sub(re.escape(basename), 'the provided information', ans, flags=re.IGNORECASE)

        # Remove formulations like "According to the information provided in '...', "
        ans = re.sub(r"According to the information provided in [\'\"].+?[\'\"],?\s*", "", ans, flags=re.IGNORECASE)

        # Remove generic leading source phrases the model might add
        ans = re.sub(r"^Based on (?:the document|the provided information|the context)[:,]?\s*", "", ans, flags=re.IGNORECASE)
    except Exception:
        pass

    return ans

