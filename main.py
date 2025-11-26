import streamlit as st
import os
import json
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any
from src.utils import load_environment_variables, log_info
# ვიყენებთ ახალ, LangChain-ისგან თავისუფალ კლასებს:
from src.vector_store import load_or_create_retriever, FaissRAGRetriever 
from src.chatbot import RAGChatbot

# --- სისტემური კონფიგურაცია ---
load_environment_variables()

# ქართული System Prompt (იურიდიული კონტექსტისთვის)
SYSTEM_PROMPT_KA = (
    "შენ ხარ პროფესიონალი სამართლებრივი დოკუმენტების ასისტენტი. შენი ამოცანაა, უპასუხო "
    "კითხვებს მოცემული სამართლებრივი დოკუმენტების საფუძველზე (კონტექსტი). "
    "პასუხი უნდა იყოს ზუსტი, ფაქტებზე დაფუძნებული და ქართულ ენაზე.\n\n"
    "ინსტრუქციები:\n"
    "1. გამოიყენე მხოლოდ მოწოდებული კონტექსტი პასუხის გენერირებისთვის.\n"
    "2. თუ მოცემულ კონტექსტში კითხვაზე პასუხი არ არსებობს, თავაზიანად უპასუხე: "
    "'სამწუხაროდ, მოცემულ დოკუმენტებში ამ კითხვაზე ზუსტი პასუხი არ მოიპოვება.'\n"
    "3. ყოველთვის მიუთითე წყაროები (Source Citation) პასუხის ბოლოს."
)


@st.cache_resource(show_spinner="⏳ ვექტორული ბაზის ჩატვირთვა/შექმნა...")
def setup_chatbot(file_paths: List[str]) -> RAGChatbot | None:
    """ამზადებს ვექტორულ ბაზას და ქმნის ჩეტბოტს."""
    try:
        # იძახებს load_or_create_retriever-ს src/vector_store.py-დან
        retriever = load_or_create_retriever(file_paths) 
        
        # იძახებს RAGChatbot-ს src/chatbot.py-დან
        chatbot = RAGChatbot(retriever, SYSTEM_PROMPT_KA)
        
        return chatbot
    except Exception as e:
        log_info(f"FATAL ERROR: ჩეტბოტის ინიციალიზაციის შეცდომა: {e}")
        st.error(f"ჩეტბოტის ინიციალიზაცია ვერ მოხერხდა. შეცდომა: {e}")
        st.caption("გთხოვთ, შეამოწმოთ ტერმინალი დეტალებისთვის და დარწმუნდით, რომ GEMINI_API_KEY დაყენებულია.")
        return None

# --- Streamlit UI ---

st.set_page_config(page_title="⚖️ Legal RAG Chatbot (Vanilla Python)")
st.title("⚖️ Legal RAG Chatbot (Gemini SDK)")
st.caption("ქართული იურიდიული დოკუმენტების ანალიზი და ჩატის მეხსიერება.")

# --- ფაილის ატვირთვის გვერდითა პანელი ---
uploaded_files = st.sidebar.file_uploader(
    "1. ატვირთეთ PDF ან TXT დოკუმენტები",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

file_paths: List[str] = []

if uploaded_files:
    # ფაილების დროებით შენახვა
    for uploaded_file in uploaded_files:
        # NamedTemporaryFile-ის გამოყენება
        try:
            with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_paths.append(tmp_file.name)
        except Exception as e:
            st.error(f"ფაილის შენახვის შეცდომა: {e}")
            continue
    
    # 2. ჩეტბოტის ინიციალიზაცია
    chatbot = setup_chatbot(file_paths)
    
    # 3. ჩატის ინტერფეისი
    if chatbot:
        st.sidebar.success("Chatbot მზადაა!")
        st.sidebar.info(f"ჩაიტვირთა {len(uploaded_files)} ფაილი.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ჩატის ისტორიის ჩვენება
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # მომხმარებლის შეტანა
        if prompt := st.chat_input("2. დასვით კითხვა დოკუმენტების შესახებ..."):
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("პასუხის გენერაცია..."):
                    try:
                        # კითხვის გაგზავნა RAGChatbot-თან
                        response = chatbot.ask_question(prompt)
                        
                        full_response = response["answer"]
                        
                        # Source Citation-ის დამატება
                        if response.get("citations"):
                            full_response += f"\n\n---\n**წყაროები (Source Citations):** {response['citations']}"
                            
                        st.markdown(full_response)
                        
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                    except Exception as e:
                        st.error(f"შეცდომა კითხვა-პასუხის დროს: {e}")
                        log_info(f"Runtime Error: {e}")
            
    # დროებითი ფაილების წაშლა (გაშვების დასრულების შემდეგ)
    # Streamlit-ის შემთხვევაში, ეს უკეთესია გაკეთდეს აპლიკაციის დახურვისას, მაგრამ დეველოპმენტისთვის ასე ვტოვებთ.
    # for path in file_paths:
    #     if os.path.exists(path):
    #         os.remove(path)

else:
    st.info("⬆️ გთხოვთ, ატვირთოთ სამართლებრივი დოკუმენტები მარცხენა პანელზე ჩატის დასაწყებად.")
    # st.stop() - ამ ხაზს არ ვიყენებთ, რათა UI არ იყოს ცარიელი