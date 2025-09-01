from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import numpy as np
import random
import io

from src.vllm_passport_recognitor import VLLMPassportRecognitor
from src.passport_recognitor import PassportRecognitor

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="static/templates")

passport_prompt_path = r"prompts/passport_prompt_gpt.md"
vllm = VLLMPassportRecognitor(passport_prompt_path)
recognitor = PassportRecognitor(vllm)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recognize-passport")
async def recognize_passport(file: UploadFile = File(...)):
    contents = await file.read()
    
    image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

    passport_fields = recognitor.extract_passport_data(image_pil)

    # passport_fields = {
    #     'passport_surname': 'ДОЛГОВ', 
    #     'passport_name': 'АЛЕКСАНДР', 
    #     'passport_patronymic': 'ПЕТРОВИЧ', 
    #     'passport_sex': 'M', 
    #     'passport_birth_date': '10.05.03', 
    #     'passport_birth_place': '1ox', 
    #     'passport_series': '7323', 
    #     'passport_number': '542628', 
    #     'passport_issue_date': '03.07.23', 
    #     'passport_department_code': '730-001', 
    #     'passport_issued_by': "GET THIS INFORMATION FROM 'depratment_code'",\
    #     "passport_citizenship": "Российсая Федерация ТЕСТ",
    #     "passport_mrz": "P<RUSDOLOV<<ALEKSANDR<PETROVICH<<<<<<<<<<<<<<<<<<\n7323542628RUS0305103M2307036<<<<<<<<<<<<<<04"
    # }

    # foreign_fields = {
    #     "foreign_doc_type": "P",
    #     "foreign_country_code": "RUS",
    #     "foreign_citizenship": "RUS",
    #     "foreign_latin_name": "IVANOV IVAN",
    #     "foreign_birth_date": "10.05.2003",
    #     "foreign_sex": "M",
    #     "foreign_birth_place": "г. Ульяновск",
    #     "foreign_number": "XXXXXXX",
    #     "foreign_issue_date": "21.02.2023",
    #     "foreign_expiry_date": "21.02.2027",
    #     "foreign_issued_by": "УМВД и ещё чего-то там...",
    #     "foreign_mrz": "<<><><>><kek<><><>lol>>><>><>><><><><<>"
    # }

    # passport_type = 'passport' if random.random() > 0.5 else 'foreign'

    # пока вернём "заглушку"
    result = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents),
        "status": "ok",
        "passport_type": passport_fields["passport_type"],
        "fields": passport_fields
    }

    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)