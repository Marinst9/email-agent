import os
import PyPDF2

def get_db():
    from app import db, KnowledgeDocument
    return db, KnowledgeDocument

def add_document(user_email, filename, text):
    from app import db, KnowledgeDocument
    
    # Избриши стари chunks од истиот документ
    KnowledgeDocument.query.filter_by(
        user_email=user_email, filename=filename
    ).delete()
    db.session.commit()
    
    # Подели на chunks
    chunks = []
    chunk_size = 500
    overlap = 50
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    
    # Зачувај во PostgreSQL
    for i, chunk in enumerate(chunks):
        doc = KnowledgeDocument(
            user_email=user_email,
            filename=filename,
            chunk_index=i,
            content=chunk
        )
        db.session.add(doc)
    db.session.commit()
    
    return len(chunks)

def search_documents(user_email, query, n_results=3):
    from app import KnowledgeDocument
    
    # Едноставно keyword пребарување
    query_words = query.lower().split()
    all_docs = KnowledgeDocument.query.filter_by(user_email=user_email).all()
    
    scored = []
    for doc in all_docs:
        content_lower = doc.content.lower()
        score = sum(1 for word in query_words if word in content_lower)
        if score > 0:
            scored.append((score, doc.content))
    
    # Сортирај по score и врати топ резултати
    scored.sort(reverse=True)
    return [content for _, content in scored[:n_results]]

def list_documents(user_email):
    from app import KnowledgeDocument
    docs = KnowledgeDocument.query.filter_by(user_email=user_email).all()
    filenames = list(set(d.filename for d in docs))
    return filenames

def delete_document(user_email, filename):
    from app import db, KnowledgeDocument
    KnowledgeDocument.query.filter_by(
        user_email=user_email, filename=filename
    ).delete()
    db.session.commit()
    return True

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text