create folder name Data and add your information into a file name info.txt


---

### Tech Stack
- **Python**
- **Streamlit** (UI)
- **FAISS** (vector storage + similarity search)
- **MiniLM (all-minilm)** — embeddings via Ollama
- **LLaMA-3 8B (llama3:8b)** — local LLM
- **Ollama** (model orchestration)
- **NumPy**, **pypdf**

---

### How It Works (Architecture)
1. Loads data/info.txt 
2. Splits it into semantic chunks  
3. Creates embeddings using MiniLM  
4. Builds a FAISS index  
5. For each user query:
   - Embeds the question  
   - Retrieves top-k most similar chunks  
   - Passes them + persona prompt into LLaMA-3  
   - Returns a grounded, personalized answer  
