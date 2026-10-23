from pydantic import BaseModel, Field
from typing import List, Optional

class CourseMapping(BaseModel):
    courseCode: str
    coNumbers: List[str]
    sourceFiles: List[str]

class QueryRequest(BaseModel):
    question: str = Field(..., example="How does HNSW graph search work?")
    courseCode: Optional[str] = Field(None, example="25CS2207E")  # Optional
    coNumber: Optional[str] = Field(None, example="CO3")          # Optional

class QueryResponse(BaseModel):
    answer: str
    matchedCourses: List[CourseMapping]
    retrievedChunksCount: int