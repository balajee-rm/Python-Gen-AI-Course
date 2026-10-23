from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from app.schemas import QueryRequest, QueryResponse
from app.rag_service import ingest_pdf_bytes, query_rag_engine

app = FastAPI(
    title="University RAG Platform Backend",
    description="Vector Search Engine with PDF Ingestion and Gemini RAG",
    version="1.0.0"
)

@app.post(
    "/api/materials/upload",
    summary="Upload & Index PDF Course Material",
    tags=["1. Material Ingestion"]
)
async def upload_pdf_material(
    courseCode: str = Form(..., example="25CS2207E"),
    coNumber: str = Form(..., example="CO3"),
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pdf files are supported for ingestion."
        )

    try:
        pdf_bytes = await file.read()
        chunk_count = ingest_pdf_bytes(
            course_code=courseCode,
            co_number=coNumber,
            file_name=file.filename,
            pdf_bytes=pdf_bytes
        )
        return {
            "status": "Success",
            "message": f"Successfully ingested {chunk_count} text chunks into vector_db.",
            "fileName": file.filename,
            "courseCode": courseCode,
            "coNumber": coNumber
        }
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post(
    "/api/student/ask",
    response_model=QueryResponse,
    summary="Ask Course Question",
    tags=["2. Search Engine"]
)
async def ask_question(request: QueryRequest):
    try:
        result = query_rag_engine(
            question=request.question,
            course_code=request.courseCode,
            co_number=request.coNumber
        )
        return QueryResponse(
            answer=result["answer"],
            matchedCourses=result["matchedCourses"],
            retrievedChunksCount=result["retrievedChunksCount"]
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))