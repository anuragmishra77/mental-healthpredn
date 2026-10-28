from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "Mental_Health_Model.pkl")

app = FastAPI(title="Mental Health Signal API", version="1.0.0")

top_countries = [
    "Other", "India", "USA", "Canada", "Australia",
    "UK", "Germany", "Mexico", "Turkey", "France"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudentData(BaseModel):
    age: int = Field(..., gt=0, le=100)
    gender: Literal["Male", "Female"]
    country: str
    academic_level: Literal["High School", "Undergraduate", "Graduate", "Postgraduate"]
    most_used_platform: Literal[
        "Facebook", "LinkedIn", "Instagram", "Twitter", "Snapchat", "TikTok",
        "YouTube", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat"
    ]
    purpose_of_use: Literal["Networking", "Education", "Entertainment", "News"]
    avg_daily_usage_hours: float = Field(..., gt=0, le=24)
    daily_unlocks: int = Field(..., gt=0)
    study_hours: float = Field(..., gt=0, le=24)
    physical_activity_hours: float = Field(..., gt=0, le=24)
    sleep_hours_per_night: float = Field(..., gt=0, le=24)
    stress_level: Literal["Low", "Medium", "High", "Very High"]


class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "message": "Mental Health Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):
    country_group = data.country if data.country in top_countries else "Other"

    input_row = pd.DataFrame([{
        "Age": data.age,
        "Gender": data.gender,
        "Country": data.country,
        "Academic_Level": data.academic_level,
        "Most_Used_Platform": data.most_used_platform,
        "Purpose_Of_Use": data.purpose_of_use,
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
        "Daily_Unlocks": data.daily_unlocks,
        "Study_Hours": data.study_hours,
        "Physical_Activity_Hours": data.physical_activity_hours,
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
        "Stress_Level": data.stress_level,
        "Grouped_country": country_group,
    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=float(prediction))


# Serve the HTML/CSS/JS from this same service so the UI and API share one URL.
# Keep this mount after API routes so /health and /predict are handled by FastAPI.
app.mount("/", StaticFiles(directory=str(BASE_DIR)), name="static")
