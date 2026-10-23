from pydantic import BaseModel, Field
from typing import List
from data.student_data import DEFAULT_STUDENTS

class CourseMark(BaseModel):
    course_name: str
    marks_obtained: int
    max_marks: int

class StudentRecord(BaseModel):
    student_name: str
    grades: List[CourseMark]

class BatchAnalysisRequest(BaseModel):
    target_student_name: str = Field(..., example="Alice Smith")
    students_database: List[StudentRecord] = Field(
        default=DEFAULT_STUDENTS,
        description="Dataset containing marks for students across courses."
    )