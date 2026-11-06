from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vision.analysis import SUPPORTED_EXERCISES, analyze_image_file, analyze_video_file


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="FormCheck",
    description="Сервис анализа техники упражнений по видео и фото",
    version="0.1.0",
)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "service": "FormCheck"}


@app.get("/api/exercises")
async def exercises() -> dict:
    return {"exercises": ["auto", *list(SUPPORTED_EXERCISES)]}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    exercise: str = Form("auto"),
) -> dict:
    file_name = file.filename or "upload"
    suffix = Path(file_name).suffix.lower()
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    normalized_exercise = (exercise or "auto").strip() or "auto"
    allowed = ["auto", *list(SUPPORTED_EXERCISES)]

    if normalized_exercise.lower().replace("_", "-") not in {item.lower().replace("_", "-") for item in allowed}:
        return {"status": "bad", "score": 0, "message": "Упражнение не поддерживается", "details": {"allowed": allowed}}

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        if suffix in image_suffixes:
            image_media_type = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
            if suffix == ".webp":
                image_media_type = "image/webp"
            result = analyze_image_file(temp_path, exercise=normalized_exercise)
            result["uploaded_image"] = f"data:{image_media_type};base64,{base64.b64encode(content).decode('ascii')}"
        else:
            result = analyze_video_file(temp_path, exercise=normalized_exercise)
            result["uploaded_image"] = ""
        result["filename"] = file_name
        return result
    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
