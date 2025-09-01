from PIL import Image
from src.vllm_passport_recognitor import VLLMPassportRecognitor
from mrz.checker.td3 import TD3CodeChecker
from paddleocr import PaddleOCR
from src.passport_recognitor import PassportRecognitor

passport_prompt_path = r"prompts/passport_prompt_gpt.md"
image_path = r"1.jpg"

if __name__ == "__main__":
    vllm = VLLMPassportRecognitor(passport_prompt_path)
    recognitor = PassportRecognitor(vllm)

    image = Image.open(image_path)
    result = recognitor.extract_passport_data(image)
    print(result)