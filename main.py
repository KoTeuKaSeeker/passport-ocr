from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import numpy as np
import random
import io
from src.passport_recognitor import PassportRecognitor

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static/templates")

passport_prompt_path = r"prompts/passport_prompt_gpt.md"
recognitor = PassportRecognitor()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recognize-passport")
async def recognize_passport(file: UploadFile = File(...)):
    contents = await file.read()
    
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image_pil)

    try:
        passport_fields = recognitor.extract_passport_data(image)
    except Exception as e:
        passport_fields = {"error": str(e)}
        
    result = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents),
        "passport_type": passport_fields.get("passport_type", "passport"),
        "fields": passport_fields
    }

    if "error" in passport_fields:
        result["error"] = str(passport_fields.get("error", None))
        del passport_fields["error"]


    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)