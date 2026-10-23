import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from schemas import BatchAnalysisRequest
from google import genai
from google.genai import types

load_dotenv()

# Retrieve Gemini configurations directly from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in the .env file.")

# Initialize the official Google GenAI client
ai_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="University Student Success Platform API",
    description=f"Gemini-powered multimodal student success platform using model: {CHAT_MODEL}",
    version="5.4.0"
)


# =====================================================================
# SCENARIO 1: Text-Based Academic Performance & Analytics
# =====================================================================
@app.post("/api/v1/student/text-analysis", summary="Scenario 1: Student Mark Analysis & Ranking")
async def analyze_student_marks(request: BatchAnalysisRequest):
    prompt = f"""
    You are an academic advisor for CSE II Year. Below is a dataset of student marks across courses.
    Target Student to Analyze: {request.target_student_name}
    Full Class Dataset: {request.model_dump_json()}

    Perform a comparative analysis for the target student and output a valid JSON response containing:
    1. "student_name": string
    2. "overall_standing": string (Choose strictly from: "Good", "Medium", or "Bad")
    3. "class_rank": integer (Calculated based on total marks compared to peers in database)
    4. "positives": list of strings (strengths based on scores)
    5. "negatives": list of strings (weaknesses or vulnerable subjects)
    6. "suggestions_to_improve": list of actionable recommendations
    """
    try:
        response = ai_client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        
        analysis_data = json.loads(response.text)

        return JSONResponse(content={
            "status": "success", 
            "model": CHAT_MODEL, 
            "analysis": analysis_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# SCENARIO 2: Image-Based Academic Diagram / Equation Analysis
# =====================================================================
@app.post("/api/v1/student/image-analysis", summary="Scenario 2: Diagram / Equation OCR & Explanation")
async def analyze_image_content(
    question: str = Form(default="Analyze this academic diagram or technical content."),
    file: UploadFile = File(...)
):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Use JPEG, PNG, or WEBP.")

    try:
        image_bytes = await file.read()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
        
        prompt = f"""
        Analyze the provided academic image/diagram considering the student prompt: '{question}'.
        Output a valid JSON response containing:
        1. "title": string (Concise title of the concept/diagram)
        2. "explanation": string (Clear breakdown of elements or equations)
        3. "similar_techniques": list of strings (Alternative methods or algorithms)
        4. "application_domains": list of strings (Practical engineering/computer science application domains)
        5. "efficiency_comparison": list of strings or objects (Comparison notes with other similar techniques)
        """
        
        response = ai_client.models.generate_content(
            model=CHAT_MODEL,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        
        image_analysis_data = json.loads(response.text)

        return JSONResponse(content={
            "status": "success", 
            "model": CHAT_MODEL, 
            "ai_response": image_analysis_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# SCENARIO 3: Audio-Based Viva Voice Answer Evaluation
# =====================================================================
@app.post("/api/v1/student/viva-evaluation", summary="Scenario 3: Viva Voice Answer Grading via Audio")
async def evaluate_viva_audio(
    viva_question: str = Form(..., example="Explain the time complexity of QuickSort in worst and average cases."),
    file: UploadFile = File(...)
):
    try:
        audio_bytes = await file.read()
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=file.content_type)
        
        prompt = f"""
        You are an internal university viva examiner. 
        The Viva Question asked is: "{viva_question}"
        Evaluate the student's audio/voice submission response.
        Output a valid JSON response containing:
        1. "transcript_summary": string
        2. "technical_accuracy_score": integer (Out of 10)
        3. "clarity_communication_score": integer (Out of 10)
        4. "final_grade": string (e.g., A+, B, C, F)
        5. "detailed_feedback": string (Constructive appraisal highlighting correct points and missing elements)
        """
        
        response = ai_client.models.generate_content(
            model=CHAT_MODEL,
            contents=[audio_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )
        
        viva_data = json.loads(response.text)

        return JSONResponse(content={
            "status": "success", 
            "model": CHAT_MODEL, 
            "evaluation": viva_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================================
# SCENARIO 4: Video-Based Mock Interview Analysis
# =====================================================================
@app.post("/api/v1/student/mock-interview-analysis", summary="Scenario 4: Mock Interview Video Assessment")
async def analyze_mock_interview_video(
    job_role: str = Form(default="Software Engineer Intern", example="Software Engineer Intern"),
    file: UploadFile = File(...)
):
    try:
        video_bytes = await file.read()
        video_part = types.Part.from_bytes(data=video_bytes, mime_type=file.content_type)
        
        prompt = f"""
        You are an expert career placement officer. Analyze this mock interview video for the role of '{job_role}'.
        Evaluate the candidate comprehensively and output a valid JSON response containing:
        1. "scores": object containing integer ratings from 1 to 10 for:
           - "technical_delivery"
           - "communication_tone"
           - "confidence_body_language"
           - "professional_presentation"
        2. "key_strengths": list of strings (What the student did well visually and verbally)
        3. "areas_of_improvement": list of strings (Mannerisms, speed of speech, or structural phrasing corrections)
        4. "verdict": string (Hireability recommendation for placement drives)
        """
        
        response = ai_client.models.generate_content(
            model=CHAT_MODEL,
            contents=[video_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        
        interview_data = json.loads(response.text)

        return JSONResponse(content={
            "status": "success", 
            "model": CHAT_MODEL, 
            "interview_report": interview_data
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))