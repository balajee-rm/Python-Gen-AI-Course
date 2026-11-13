from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv()
import jwt
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# MongoDB Cloud Connection URL
MONGO_DETAILS = "mongodb+srv://admin:admin@cluster0.cjqwipx.mongodb.net/"
client = AsyncIOMotorClient(MONGO_DETAILS)
users_collection = client.mydb.users

# Single Schema for Signup and Signin
class User(BaseModel):
    username: str
    password: str

# Helper: Fixed 30-min token with merged return (stores under "user" key)
def create_access_token(data: dict) -> str:
    return jwt.encode({**data, "exp": datetime.utcnow() + timedelta(minutes=30)}, SECRET_KEY, algorithm=ALGORITHM)

# Helper: Direct header extraction for a custom "token" field
async def get_current_user(token: str = Header(...)) -> str:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])["user"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.post("/signup")
async def signup(user: User):
    if await users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = pwd_context.hash(user.password)
    await users_collection.insert_one({"username": user.username, "password": hashed_password})
    return JSONResponse(content={"message": "User registered successfully"}, status_code=200)

@app.post("/signin")
async def signin(user: User):
    db_user = await users_collection.find_one({"username": user.username})
    if not db_user or not pwd_context.verify(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    token = create_access_token({"user": user.username})
    return JSONResponse(content={"access_token": token}, status_code=200)

@app.get("/protected")
async def protected_route(username: str = Depends(get_current_user)):
    return JSONResponse(content={"message": f"Hello {username}, token is valid!"}, status_code=200)