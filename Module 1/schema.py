from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    topic: str = Field(..., description="The topic for which to generate text.", example="Neural Networks")
    subject: str = Field(..., description="The subject related to the topic.", example="Artificial Intelligence")
    student_level: str = Field(..., description="The educational level of the student.", example="beginner")