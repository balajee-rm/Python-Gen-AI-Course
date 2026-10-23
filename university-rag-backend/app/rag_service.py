import os
import io
import chromadb
from pypdf import PdfReader
from collections import defaultdict
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Read model configurations from .env with fallback defaults
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")

# Initialize GenAI Client using google-genai SDK
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB Vector DB
chroma_client = chromadb.PersistentClient(path="./vector_db")

collection = chroma_client.get_or_create_collection(
    name="university_course_materials",
    metadata={"hnsw:space": "cosine"}
)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def ingest_pdf_bytes(course_code: str, co_number: str, file_name: str, pdf_bytes: bytes) -> int:
    """Extracts text from PDF bytes, generates embeddings, and indexes chunks into ChromaDB."""
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"

    if not extracted_text.strip():
        raise ValueError(f"Could not extract readable text from {file_name}. File may be scanned images or empty.")

    chunks = chunk_text(extracted_text)

    for idx, chunk in enumerate(chunks):
        embed_resp = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk
        )
        vector = embed_resp.embeddings[0].values

        doc_id = f"{course_code}_{co_number}_{file_name}_chunk_{idx}"
        
        collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[chunk],
            metadatas=[{
                "courseCode": course_code,
                "coNumber": co_number,
                "fileName": file_name,
                "chunkIndex": idx
            }]
        )
        
    return len(chunks)

def query_rag_engine(question: str, course_code: str = None, co_number: str = None) -> dict:
    """Queries vector DB with optional filters ($and handling) and synthesizes answer via Gemini."""
    embed_resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question
    )
    query_vector = embed_resp.embeddings[0].values

    # Build ChromaDB filter clause
    conditions = []
    if course_code and course_code.strip():
        conditions.append({"courseCode": course_code.strip()})
    if co_number and co_number.strip():
        conditions.append({"coNumber": co_number.strip()})

    where_clause = None
    if len(conditions) == 1:
        where_clause = conditions[0]
    elif len(conditions) > 1:
        where_clause = {"$and": conditions}

    query_params = {
        "query_embeddings": [query_vector],
        "n_results": 5
    }
    if where_clause:
        query_params["where"] = where_clause

    results = collection.query(**query_params)

    retrieved_docs = results['documents'][0] if results['documents'] else []
    retrieved_meta = results['metadatas'][0] if results['metadatas'] else []

    if not retrieved_docs:
        return {
            "answer": "No relevant course material found for your question.",
            "matchedCourses": [],
            "retrievedChunksCount": 0
        }

    course_data = defaultdict(lambda: {"cos": set(), "files": set()})
    context_builder = ""
    
    for doc, meta in zip(retrieved_docs, retrieved_meta):
        c_code = meta['courseCode']
        co_num = meta['coNumber']
        f_name = meta['fileName']
        
        course_data[c_code]["cos"].add(co_num)
        course_data[c_code]["files"].add(f_name)
        
        context_builder += f"[Course: {c_code} | CO: {co_num} | File: {f_name}]\n{doc}\n\n"

    matched_courses = [
        {
            "courseCode": code,
            "coNumbers": list(data["cos"]),
            "sourceFiles": list(data["files"])
        }
        for code, data in course_data.items()
    ]

    system_prompt = f"""
    You are an AI University Assistant. 
    Answer the student's question strictly using the retrieved course material below.
    Explicitly cite the Course Code and CO/Module in your response.

    Context Material:
    {context_builder}

    Student Question: {question}
    """

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=system_prompt
    )

    return {
        "answer": response.text,
        "matchedCourses": matched_courses,
        "retrievedChunksCount": len(retrieved_docs)
    }