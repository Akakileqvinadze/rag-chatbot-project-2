import os
from typing import List
from pypdf import PdfReader
from src.utils import log_info

# LangChain-ის Document-ის იმიტაცია (უბრალო Dict-ით)
Document = dict

def load_documents(file_paths: List[str]) -> List[Document]:
    """ტვირთავს დოკუმენტებს (PDF/TXT) და აბრუნებს Dict-ების სიას (content, metadata)."""
    all_documents = []
    
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        log_info(f"დოკუმენტის ჩატვირთვა: {file_path}")
        
        try:
            if ext == '.pdf':
                # PyPDFLoader-ის ლოგიკის ხელით იმპლემენტაცია
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    all_documents.append(
                        {
                            "page_content": page.extract_text() or "",
                            "metadata": {"source": os.path.basename(file_path), "page": i + 1}
                        }
                    )
            elif ext == '.txt':
                with open(file_path, 'r', encoding="utf-8") as f:
                    all_documents.append(
                        {
                            "page_content": f.read(),
                            "metadata": {"source": os.path.basename(file_path), "page": "N/A"}
                        }
                    )
            else:
                log_info(f"ფაილის ფორმატი '{ext}' არ არის მხარდაჭერილი.")

            log_info(f"ჩაიტვირთა {len(all_documents)} გვერდი/სექცია {os.path.basename(file_path)}-დან.")
            
        except Exception as e:
            log_info(f"შეცდომა დოკუმენტის ჩატვირთვისას {file_path}: {e}")

    return all_documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    დოკუმენტების ნაწილებად დაყოფა (chunking).
    LangChain-ის RecursiveCharacterTextSplitter-ის ლოგიკის გამოყენება.
    """
    log_info("დოკუმენტების ნაწილებად დაყოფა (chunking)...")
    
    chunks = []
    chunk_size = 1000
    chunk_overlap = 200
    
    for doc in documents:
        content = doc['page_content']
        metadata = doc['metadata']
        
        # უმარტივესი სფლითინგი (შეიძლება გართულდეს regex-ით, მაგრამ სტაბილურობისთვის)
        # ვინაიდან LangChain-ის სფლითერი აღარ გვაქვს, გამოვიყენოთ მარტივი ტექსტის დაყოფა
        current_idx = 0
        while current_idx < len(content):
            end_idx = min(current_idx + chunk_size, len(content))
            chunk_content = content[current_idx:end_idx]
            
            chunks.append({
                "chunk_content": chunk_content,
                "metadata": metadata.copy()
            })
            
            # გადაფარვის ლოგიკა
            current_idx += chunk_size - chunk_overlap
            if current_idx >= len(content):
                break
                
    log_info(f"საერთო ჯამში შეიქმნა {len(chunks)} ნაწილი (chunks).")
    return chunks