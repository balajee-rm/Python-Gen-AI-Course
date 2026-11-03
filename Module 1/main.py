import os
import json

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from dotenv import load_dotenv
from schema import TextRequest

load_dotenv()

genai_api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("CHAT_MODEL")

app = FastAPI(
    title="AI Tutor",
    description=f"AI Powerd by {model}",
    version="1.0.0"
)

if not genai_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
if not model:
    raise ValueError("CHAT_MODEL is not set in the environment variables.")

gemini_client = genai.Client(api_key=genai_api_key)

# Text input to generate explanation for the topic
@app.post("/generate-explanation")
async def generate_explanation(request: TextRequest):
    prompt = f"""
    You are an AI tutor creating study material for a student.
    Topic: "{request.topic}"
    Subject: "{request.subject or "Not specified"}"
    Student level: "{request.student_level}"

    Explain the topic clearly and accurately using language suited to the student.
    Output a valid JSON response containing:
    1. "topic": string
    2. "lesson_content": string (clear explanation for study)
    3. "key_terms": list of objects with "term" and "definition" strings
    4. "example": string (a relevant example)
    5. "practice_questions": list of 3 questions
    """

    try:
        response = gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        # Parse the response text as JSON
        response_json = json.loads(response.text)
        return JSONResponse(content=response_json)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Academic Image Analysis and Explanation
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), question: str = Form(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        return JSONResponse(content={"error": "Invalid file type. Only JPEG and PNG images are allowed."}, status_code=400)

    try:
        # Read the uploaded image file
        image_bytes = await file.read()

        image_input = types.Part.from_bytes(data = image_bytes, mime_type = file.content_type)

        # Create a prompt for the AI model
        prompt = f"""
        You are an AI tutor analyzing an academic image.
        Question: "{question}"

        Analyze the image and provide a clear explanation suitable for the student.
        Output a valid JSON response containing:
        1. "topic": string
        2. "image_analysis": string (clear explanation of the image)
        3. "key_terms": list of objects with "term" and "definition" strings
        4. "example": string (a relevant example)
        5. "practice_questions": list of 3 questions
        """

        response = gemini_client.models.generate_content(
            model=model,
            contents=[image_input, prompt],
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        
        # Parse the response text as JSON
        response_json = json.loads(response.text)
        return JSONResponse(content=response_json)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Text Question and Audio Answer Evaluation
@app.post("/evaluate-answer")
async def evaluate_answer(question: str = Form(...), answer: UploadFile = File(...)):

    if answer.content_type not in ["audio/mpeg", "audio/wav", "audio/mp3"]:
        return JSONResponse(content={"error": "Invalid file type. Only MP3 and WAV audio files are allowed."}, status_code=400)

    try:
        # Read the uploaded audio file
        audio_bytes = await answer.read()

        audio_input = types.Part.from_bytes(data = audio_bytes, mime_type = answer.content_type)

        # Create a prompt for the AI model
        prompt = f"""
        You are an AI tutor evaluating a student's answer to a question.
        Question: "{question}"

        Evaluate the student's answer provided in the audio file.
        Output a valid JSON response containing:
        1. "question": string
        2. "evaluation": string (feedback on the answer)
        3. "score": integer (0-100)
        4. "suggestions": list of strings (improvement suggestions)
        """

        response = gemini_client.models.generate_content(
            model=model,
            contents=[audio_input, prompt],
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        
        # Parse the response text as JSON
        response_json = json.loads(response.text)
        return JSONResponse(content=response_json)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Video Analysis and Generating 3 MCQs Questions with Answers
@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):

    if file.content_type not in ["video/mp4", "video/avi", "video/mov"]:
        return JSONResponse(content={"error": "Invalid file type. Only MP4, AVI, and MOV video files are allowed."}, status_code=400)

    try:
        # Read the uploaded video file
        video_bytes = await file.read()

        video_input = types.Part.from_bytes(data = video_bytes, mime_type = file.content_type)

        # Create a prompt for the AI model
        prompt = f"""
        You are an AI tutor analyzing an academic video.

        Provide a clear and accurate analysis of the video in relation to the question.
        Generate 3 multiple-choice questions (MCQs) based on the video content, each with 4 options and the correct answer.
        Output a valid JSON response containing:
        1. "video_analysis": string (detailed analysis of the video)
        2. "mcq_questions": list of objects with "question", "options" (list of strings), and "correct_answer" (string)
        """

        response = gemini_client.models.generate_content(
            model=model,
            contents=[video_input, prompt],
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json"
            )
        )
        
        # Parse the response text as JSON
        response_json = json.loads(response.text)
        return JSONResponse(content=response_json)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)