from typing import List
import re
import unicodedata
from paddleocr import PaddleOCR
from rapidfuzz import fuzz


class PassportIdentifier:
    def __init__(self):
        self.foreign_passport_keywords = [
            "FEDERATION","PASSPORT","Surname","Nationality","Place of birth",
            "Type","Code of issuing State","Passport No.","Date of expiry","Date of issue",
            "Sex","Authority","Holder's signature","Given names"
        ]
        self.foreign_passport_keywords = [t.lower() for t in self.foreign_passport_keywords]


    def identify_passport_type(self, recognized_texts: List[str], threshold: float = 80) -> str:
        """
        Returns 'foreign' if any recognized OCR token (or its parts) matches English passport keywords.
        Otherwise, returns 'internal'.
        """
        if not recognized_texts:
            return "internal"

        for text in recognized_texts:
            text = text.lower()

            if len(text) < 3:
                continue

            ocr_tokens = text.split()
            if not ocr_tokens:
                continue

            ocr_tokens = [text] + ocr_tokens

            for keyword in self.foreign_passport_keywords:
                for ocr_tok in ocr_tokens:
                    if len(ocr_tok) < 3:
                        continue
                    score = fuzz.ratio(ocr_tok, keyword)
                    if score >= threshold:
                        return "foreign"

        return "internal"

# --- Example usage ---
if __name__ == "__main__":

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='ru')
    
    pid = PassportIdentifier()

    test_data = [
        {"path": "data/images/1.png", "type": "internal"},
        {"path": "data/images/2.jpg", "type": "internal"},
        {"path": "data/images/3.jpg", "type": "internal"},
        {"path": "data/images/danil.jpg", "type": "internal"},
        {"path": "data/images/sasha_2.jpg", "type": "internal"},
        {"path": "data/images/sasha_3.png", "type": "internal"},
        {"path": "data/images/sasha.jpg", "type": "internal"},
        {"path": "data/2312660261.jpg", "type": "foreign"},
        {"path": "data/passport-inner.png", "type": "foreign"}
    ]


    for entry in test_data:
        path, passport_type = entry["path"], entry["type"]

        result = ocr.predict(path)

        identified_type = pid.identify_passport_type(result[0]["rec_texts"])

        correctness = "✅" if identified_type == passport_type else "❌"

        print(f"{correctness} (correct type: {passport_type})   path: {path}")
    