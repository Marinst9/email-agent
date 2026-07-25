import os
import chromadb
from chromadb.utils import embedding_functions
import PyPDF2
import json

# ChromaDB клиент
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

def get_collection(user_email):
    """Земи или создај колекција за корисникот"""
    safe_name = user_email.replace('@', '_').replace('.', '_')
    return chroma_client.get_or_create_collection(
        name=f"docs_{safe_name}",
        embedding_function=embedding_fn
    )

def add_document(user_email, filename, text):
    """Додај документ во базата на знаење"""
    collection = get_collection(user_email)
    
    # Подели го текстот на чанкови од 500 карактери
    chunks = []
    chunk_size = 500
    overlap = 50
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    
    # Додај ги чанковите во ChromaDB
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{filename}_{i}"],
            metadatas=[{"filename": filename, "chunk": i}]
        )
    
    return len(chunks)

def search_documents(user_email, query, n_results=3):
    """Пребарај релевантни документи за прашањето"""
    try:
        collection = get_collection(user_email)
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count())
        )
        if results and results['documents']:
            return results['documents'][0]
        return []
    except Exception as e:
        print(f"RAG грешка: {e}")
        return []

def delete_document(user_email, filename):
    """Избриши документ"""
    try:
        collection = get_collection(user_email)
        results = collection.get(where={"filename": filename})
        if results['ids']:
            collection.delete(ids=results['ids'])
        return True
    except Exception as e:
        print(f"Грешка при бришење: {e}")
        return False

def list_documents(user_email):
    """Листа на документи за корисникот"""
    try:
        collection = get_collection(user_email)
        results = collection.get()
        filenames = set()
        if results['metadatas']:
            for meta in results['metadatas']:
                filenames.add(meta['filename'])
        return list(filenames)
    except:
        return []

def extract_text_from_pdf(file_path):
    """Извади текст од PDF"""
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text