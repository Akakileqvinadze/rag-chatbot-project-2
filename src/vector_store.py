import os
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.utils import log_info
from src.document_processor import load_documents, split_documents

# ვექტორული ბაზის შენახვის გზა
VECTOR_DB_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.json" # FAISS-ს სჭირდება შესაბამისი ტექსტი
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

class FaissRAGRetriever:
    """FAISS-ზე დაფუძნებული რიტრივერი Sentence-Transformers-ით."""

    def __init__(self, faiss_index, chunks: List[Dict]):
        self.index = faiss_index
        self.chunks = chunks
        
    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """პოულობს K უახლოეს ნაწილს (chunks) ვექტორულ სივრცეში."""
        # FAISS-ის ძიება: D - დისტანცია, I - ინდექსი
        D, I = self.index.search(query_embedding, k)
        
        results = []
        for idx in I[0]:
            if idx != -1: # დარწმუნდით, რომ ინდექსი ვალიდურია
                results.append(self.chunks[idx])
        return results

# ახალი კოდი:
def get_embeddings_model() -> SentenceTransformer:
    """აინიცირებს მულტილინგვურ Embeddings მოდელს."""
    log_info("Embeddings მოდელის ინიციალიზაცია...") # უბრალოდ ლოგირება
    return SentenceTransformer(MODEL_NAME)

def load_or_create_retriever(file_paths: List[str]) -> FaissRAGRetriever:
    """ტვირთავს ან ქმნის FAISS ინდექსს და რიტრივერს."""
    model = get_embeddings_model()
    
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(CHUNKS_PATH):
        log_info("FAISS ინდექსი და ჩანქები ნაპოვნია. მიმდინარეობს ჩატვირთვა.")
        
        # 1. ინდექსის ჩატვირთვა
        faiss_index = faiss.read_index(VECTOR_DB_PATH)
        
        # 2. ჩანქების ჩატვირთვა (ჩვენ გვჭირდება ტექსტი)
        import json
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        log_info("Retriever წარმატებით შეიქმნა არსებული ინდექსის საფუძველზე.")
        return FaissRAGRetriever(faiss_index, chunks)
    
    else:
        log_info("FAISS ინდექსი ვერ მოიძებნა. მიმდინარეობს შექმნა.")
        documents = load_documents(file_paths)
        if not documents:
            raise ValueError("დოკუმენტები ვერ ჩაიტვირთა.")
            
        chunks = split_documents(documents)
        
        # 3. ვექტორების გენერაცია
        texts = [chunk['chunk_content'] for chunk in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype('float32') # FAISS-ს სჭირდება float32
        
        # 4. FAISS ინდექსის შექმნა
        d = embeddings.shape[1] # ვექტორის განზომილება
        faiss_index = faiss.IndexFlatL2(d) # L2 დისტანციის ინდექსი
        faiss_index.add(embeddings) # ვექტორების დამატება
        
        # 5. ინდექსის და ჩანქების შენახვა
        faiss.write_index(faiss_index, VECTOR_DB_PATH)
        import json
        with open(CHUNKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        log_info(f"FAISS ინდექსი და ჩანქები წარმატებით შენახულია.")
        return FaissRAGRetriever(faiss_index, chunks)

def get_retriever_model() -> SentenceTransformer:
    """აბრუნებს embeddings მოდელს, რომელიც საჭიროა run-time-ისთვის."""
    return SentenceTransformer(MODEL_NAME)