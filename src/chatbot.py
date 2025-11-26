from typing import Dict, Any, List
# ვცვლით genai-ს იმპორტს სწორი გზით და ვარქმევთ მას Client-ს
from google.genai import Client as GenaiClient # <--- ეს აფიქსირებს GenaiClient-ის შეცდომას
from google.genai import types
from src.utils import get_gemini_api_key, log_info
from src.vector_store import FaissRAGRetriever, get_retriever_model
import numpy as np

class RAGChatbot:
    """RAG Chatbot სისტემა Gemini SDK-ით და ხელით მეხსიერების მართვით."""

    def __init__(self, retriever: FaissRAGRetriever, system_prompt: str):
        self.retriever = retriever
        self.embed_model = get_retriever_model()
        self.system_prompt = system_prompt
        self.chat_history = [] # Conversation memory
        self.client = self._initialize_client()
        log_info("RAG Chatbot წარმატებით შეიქმნა (Vanilla Python).")

    def _initialize_client(self):
        """აინიცირებს Google GenAI კლიენტს."""
        api_key = get_gemini_api_key()
        if not api_key:
             raise ValueError("API Key არ არის დაყენებული.")
             
        # ახლა GenaiClient სწორად არის იმპორტირებული
        return GenaiClient(api_key=api_key) 
    
    # src/chatbot.py - დაამატეთ ეს RAGChatbot კლასში

    def _build_context_prompt(self, context: List[Dict], query: str) -> tuple[str, set]:
        """აკონსტრუირებს საბოლოო Prompt-ს კონტექსტის, ისტორიისა და კითხვის საფუძველზე."""
        
        # 1. კონტექსტის ფორმატირება წყაროს მითითებით
        context_str = "მოცემული სამართლებრივი კონტექსტი:\n\n"
        citations = []
        for i, doc in enumerate(context):
            source = doc['metadata'].get('source', 'უცნობი დოკუმენტი')
            page = doc['metadata'].get('page', 'N/A')
            citation = f"{source} (გვ. {page})"
            context_str += f"--- კონტექსტი {i+1} ({citation}) ---\n{doc['chunk_content']}\n\n"
            citations.append(citation)
        
        # 2. ისტორიის ფორმატირება
        history_str = "\n".join([f"მომხმარებელი: {item['question']}\nასისტენტი: {item['answer']}" for item in self.chat_history])
        
        # 3. საბოლოო Prompt
        final_prompt = f"""
        {self.system_prompt}

        საუბრის ისტორია:
        {history_str or "საუბრის ისტორია არ არსებობს."}

        {context_str}

        მომხმარებლის მიმდინარე კითხვა: {query}
        """
        
        return final_prompt, set(citations)

    def ask_question(self, question: str) -> Dict[str, Any]:
        """სვამს კითხვას, იძიებს, გენერირებს პასუხს და აახლებს მეხსიერებას."""
        
        # 1. რიტრივალი: კითხვის ვექტორიზაცია
        query_embedding = self.embed_model.encode([question], convert_to_numpy=True).astype('float32')
        retrieved_docs = self.retriever.retrieve(query_embedding)
        
        # 2. Prompt-ის შექმნა
        prompt_text, citations_set = self._build_context_prompt(retrieved_docs, question)
        
        # 3. პასუხის გენერაცია Gemini-ის გამოყენებით
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt_text],
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            answer = response.text
            
            # 4. მეხსიერების განახლება
            self.chat_history.append({"question": question, "answer": answer})
            
            # 5. პასუხის დამუშავება
            citations_str = "; ".join(citations_set)
            
            return {
                "answer": answer,
                "citations": citations_str
            }
            
        except Exception as e:
            log_info(f"შეცდომა Gemini API-ის გამოძახებისას: {e}")
            return {
                "answer": "ბოდიშს გიხდით, API-ის გამოძახებისას ტექნიკური შეცდომა მოხდა.",
                "citations": ""
            }