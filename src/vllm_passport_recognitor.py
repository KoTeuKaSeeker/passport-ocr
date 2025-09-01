import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

class VLLMPassportRecognitor():
    def __init__(self, passport_prompt_path: str):
        model_id = "openbmb/MiniCPM-V-4_5"

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder="offload",
            low_cpu_mem_usage=True
        )

        self.model.config.use_cache = False # Для экономии памяти
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        with open(passport_prompt_path, "r", encoding="utf-8") as file:
            self.passport_prompt = file.read()
    
    def recognize_passport(self, image: Image.Image) -> dict:
        msgs = [
            {"role": "user", "content": self.passport_prompt},
            {"role": "user", "content": [image, "Определи тип документа и верни JSON согласно схеме. Отвечай ТОЛЬКО JSON в соответствии со схемой (если не паспорт — вернуть поле error)"]}
        ]
        answer = self.model.chat(msgs=msgs, tokenizer=self.tokenizer, enable_thinking=False)

        obj = json.loads(answer)
        return obj