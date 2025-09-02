from mrz.checker.td3 import TD3CodeChecker
from paddleocr import PaddleOCR
import re
from datetime import datetime
from typing import List, Optional
import numpy as np
import math
from src.vllm_passport_recognitor import VLLMPassportRecognitor
from PIL import Image
import json
import cv2


class PassportRecognitor:
    ocr: PaddleOCR

    def __init__(self, vllm: VLLMPassportRecognitor):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='ru')
        
        self.vllm = vllm
        
        self.DATE_CHARS_RE = re.compile(r'[0-9OQIDIl\|SsZzBbGg,./\\\-\s]{4,12}')

        self.OCR_MAP = {
            'O':'0','o':'0','Q':'0','D':'0',
            'I':'1','l':'1','i':'1','|':'1',
            'S':'5','s':'5',
            'Z':'2','z':'2',
            'B':'8','b':'6',
            'G':'6','g':'9',
            ',':'.','/':'.','\\':'.','-':'.',' ':'.'
        }
        
        self.MRZ_TO_CYR = {
            'A':'А',
            'B':'Б',
            'V':'В',
            'G':'Г',
            'D':'Д',
            'E':'Е',
            'Z':'З',
            'I':'И',
            'K':'К',
            'L':'Л',
            'M':'М',
            'N':'Н',
            'O':'О',
            'P':'П',
            'R':'Р',
            'S':'С',
            'T':'Т',
            'U':'У',
            'F':'Ф',
            'H':'Х',
            'C':'Ц',
            'J':'Ж',
            'Y':'Ы',
            'Q':'Й',
            'X':'Ъ',
            # MRZ special digits / symbols used by Russian MRZ:
            '2':'Ё',  # Ё → 2
            '3':'Ч',  # Ч → 3
            '4':'Ш',  # Ш → 4
            '6':'Э',  # Э → 6
            '9':'Ь',  # ь → 9
            'W':'Щ',  # Щ → W
            # filler and separators
            '<': ' ', 
        }
        
    def expand_year_two_digits(self, yy_str: str) -> int:
        """Expand two-digit year to four digits using cutoff by current year.
        If yy <= current_year%100 -> 2000+yy else 1900+yy."""
        try:
            yy = int(yy_str)
        except Exception:
            return None
        current_year = 2025  # явный год (по заданию: текущая дата 2025-09-01)
        cutoff = current_year % 100
        return 2000 + yy if yy <= cutoff else 1900 + yy

    def format_date_from_yymmdd(self, yymmdd: str) -> str:
        """Convert YYMMDD (MRZ) to dd.mm.YYYY, return '' on failure."""
        if not yymmdd or len(yymmdd) != 6:
            return ""
        yy = yymmdd[0:2]
        mm = yymmdd[2:4]
        dd = yymmdd[4:6]
        full_year = self.expand_year_two_digits(yy)
        if full_year is None:
            return ""
        # basic validation
        try:
            d = datetime(year=full_year, month=int(mm), day=int(dd))
        except Exception:
            return ""
        return f"{dd}.{mm}.{full_year}"
        
    def extract_passport_data(self, image: Image.Image):
        image = image.convert("RGB")
        np_image = np.array(image)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        prompt = """
Определи, является ли паспорт загранпаспортом или обычным паспортом. Это можно
сделать по следующей черте. У загранпаспорта имя гражданина написанно сразу
как на русском, так и на английском (сначала на русском, затем через слеш "/" на
английском), тоже самое и с фамилией. В обычном же паспорте имя и фамилия пишется
чисто на русском.

Определи место рождения гражданина, а также то, кем был выдан паспорт.
Верни результат в виде JSON. Тип паспорта запиши в поле "passport_type" - если
это обычный российский паспорт, то запиши в это поле строку "passport", если же
загранпасспорт - то строку "foreign". Место рождения запиши в поле "birth_place",
а то, кем был выдан паспорт запиши в поле "issued_by".
"""

        msgs = [
            {"role": "user", "content": [image, prompt]}
        ]
        answer = self.vllm.model.chat(msgs=msgs, tokenizer=self.vllm.tokenizer, enable_thinking=False)
        obj = json.loads(answer)

        output_dict: dict = {}
        passport_type = obj.get("passport_type", "")
        output_dict["passport_type"] = passport_type

        output_dict[f"{passport_type}_birth_place"] = obj.get("birth_place", "")
        output_dict[f"{passport_type}_issued_by"] = obj.get("issued_by", "")

        result = self.ocr.predict(np_image)

        mrz_first_line = next((text for text in result[0]["rec_texts"] if text.endswith("<<<")), None)
        mrz_second_line = next((text for text in result[0]["rec_texts"] if text != mrz_first_line and "<<<" in text), None)

        if not mrz_first_line or not mrz_second_line:

            output_dict["error"] = "MRZ not found"
            return output_dict
        else:
            mrz_raw = mrz_first_line + "\n" + mrz_second_line
            try:
                td3_check = TD3CodeChecker(mrz_raw)
                fields = td3_check.fields()
            except Exception as e:
                output_dict["error"] = f"MRZ parsing failed: {e}"
                return output_dict

            def fget(name: str) -> str:
                return getattr(fields, name, "") or ""

            latin_surname = fget("surname")
            latin_name_block = fget("name")

            def clean_name_block(nb: str) -> str:
                if not nb:
                    return ""
                s = nb.replace("<", " ").strip()

                s = s.replace("0", "O").replace("1", "I")
                s = " ".join(s.split())
                return s

            clean_names = clean_name_block(latin_name_block)
            name_parts = clean_names.split(" ") if clean_names else []

            output_dict[f"{passport_type}_surname"] = self.mrz_name_to_cyrillic(latin_surname) if latin_surname else ""
            output_dict[f"{passport_type}_name"] = self.mrz_name_to_cyrillic(name_parts[0]) if len(name_parts) >= 1 else ""
            output_dict[f"{passport_type}_patronymic"] = self.mrz_name_to_cyrillic(name_parts[1]) if len(name_parts) >= 2 else ""

            output_dict[f"{passport_type}_sex"] = fget("sex") or ""
            output_dict[f"{passport_type}_birth_date"] = self.format_date_from_yymmdd(fget("birth_date")) if fget("birth_date") else ""

            if passport_type == "passport":
                docnum = fget("document_number")
                opt = fget("optional_data")
                if docnum and len(docnum) >= 6 and opt:
                    output_dict["passport_series"] = docnum[0:3] + (opt[0] if len(opt) >= 1 else "")
                    output_dict["passport_number"] = docnum[3:]
                else:
                    if docnum and len(docnum) >= 6:
                        output_dict["passport_series"] = docnum[0:3]
                        output_dict["passport_number"] = docnum[3:]
                    else:
                        output_dict["passport_series"] = ""
                        output_dict["passport_number"] = ""

                if opt and len(opt) >= 7:
                    issue_yymmdd = opt[1:7]
                    output_dict[f"{passport_type}_issue_date"] = self.format_date_from_yymmdd(issue_yymmdd)
                else:
                    output_dict[f"{passport_type}_issue_date"] = obj.get("issue_date", "")

                if opt and len(opt) >= 13:
                    output_dict[f"{passport_type}_department_code"] = opt[7:10] + "-" + opt[10:13]
                else:
                    output_dict[f"{passport_type}_department_code"] = ""

            elif passport_type == "foreign":
                foreign_fields = {
                    "foreign_doc_type": fget("document_type") or "P",
                    "foreign_country_code": fget("country") or "",
                    "foreign_citizenship": fget("nationality") or "",
                    "foreign_latin_name": "",
                    "foreign_birth_date": self.format_date_from_yymmdd(fget("birth_date")) if fget("birth_date") else "",
                    "foreign_sex": fget("sex") or "",
                    "foreign_birth_place": obj.get("birth_place", ""),
                    "foreign_number": fget("document_number") or "",
                    "foreign_issue_date": "",
                    "foreign_expiry_date": self.format_date_from_yymmdd(fget("expiry_date")) if fget("expiry_date") else "",
                    "foreign_issued_by": obj.get("issued_by", ""),
                    "foreign_mrz": mrz_raw,
                }

                if latin_surname and len(name_parts) >= 1 and name_parts[0]:
                    foreign_fields["foreign_latin_name"] = f"{latin_surname} {name_parts[0]}"
                else:
                    foreign_fields["foreign_latin_name"] = (latin_surname + " " + clean_names).strip()

                opt = fget("optional_data")
                if opt and len(opt) >= 7:
                    issue_yymmdd = opt[1:7]
                    parsed_issue = self.format_date_from_yymmdd(issue_yymmdd)
                    foreign_fields["foreign_issue_date"] = parsed_issue

                if not foreign_fields["foreign_issue_date"]:
                    foreign_fields["foreign_issue_date"] = obj.get("issue_date", "")

                mrz_valid = None
                mrz_warnings = None
                if hasattr(td3_check, "is_valid") and callable(getattr(td3_check, "is_valid")):
                    try:
                        mrz_valid = td3_check.is_valid()
                    except Exception:
                        mrz_valid = None
                if mrz_valid is None and hasattr(td3_check, "valid"):
                    mrz_valid = getattr(td3_check, "valid", None)

                if hasattr(td3_check, "warnings"):
                    try:
                        mrz_warnings = td3_check.warnings
                    except Exception:
                        mrz_warnings = None

                foreign_fields["foreign_mrz_valid"] = mrz_valid
                foreign_fields["foreign_mrz_warnings"] = mrz_warnings

                output_dict.update(foreign_fields)

            else:
                output_dict["error"] = "Unknown passport_type"

        return output_dict

    def mrz_name_to_cyrillic(self, mrz_name_token: str) -> str:
        """Convert a single MRZ name token (like 'IVANOV<<ALEKSEQ<<<<<<<<') to Cyrillic string."""
        # normalize: MRZ is uppercase A-Z, digits and '<'
        s = mrz_name_token.strip().upper()
        # replace each char via map; unknown char left as-is (or could be '?')
        chars = []
        for ch in s:
            if ch in self.MRZ_TO_CYR:
                chars.append(self.MRZ_TO_CYR[ch])
            # elif 'A' <= ch <= 'Z':
            #     # map straightforward Latin consonants/vowels where MRZ uses simple A->А etc.
            #     # a few letters already in MRZ_TO_CYR; remaining map A->А, B->Б etc. (some already mapped)
            #     # fallback: try direct single-letter mapping via a minimal mapping:
            #     fallback = {
            #         'B':'Б','C':'Ц','D':'Д','E':'Е','G':'Г','H':'Х','I':'И','J':'Ж','K':'К',
            #         'L':'Л','M':'М','N':'Н','O':'О','P':'П','Q':'Й','R':'Р','S':'С','T':'Т',
            #         'U':'У','V':'В','W':'Щ','X':'Ъ','Y':'Ы','Z':'З','A':'А','F':'Ф'
            #     }
            #     chars.append(fallback.get(ch, ch))
            else:
                chars.append(ch)
        # collapse multiple spaces and trim
        result = ' '.join(''.join(chars).split())
        return result

    def lines_intersect_distance(self, line1, line2):
        p1, q1 = line1
        p2, q2 = line2

        # Convert coordinates to floats to avoid overflow
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(q1[0]), float(q1[1])
        x3, y3 = float(p2[0]), float(p2[1])
        x4, y4 = float(q2[0]), float(q2[1])

        # Compute denominator
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom == 0:
            return None  # parallel or colinear

        # Intersection point
        Px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        Py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

        inter_point = [Px, Py]

        # Check if intersection lies on both segments
        def on_segment(p, q, r):
            return (min(p[0], r[0]) - 1e-6 <= q[0] <= max(p[0], r[0]) + 1e-6 and
                    min(p[1], r[1]) - 1e-6 <= q[1] <= max(p[1], r[1]) + 1e-6)

        if on_segment(p1, inter_point, q1) and on_segment(p2, inter_point, q2):
            dx = Px - x1
            dy = Py - y1
            return math.hypot(dx, dy)
        else:
            return None
    
    def normalize_vector(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:  # to avoid division by zero
            return v
        return v / norm


    def find_date_index(self, lines: List[str]) -> int:
        """
        Returns the index of the element in `lines` that looks like a date, or -1 if none found.
        Assumes there's at most one real date in the list (but will return the first good match).
        """
        found_indices = []
        for idx, line in enumerate(lines):

            # scan for date-like substrings in the line
            for m in self.DATE_CHARS_RE.finditer(line):
                token = m.group().strip()
                if len(re.sub(r'\D','', token)) < 4:
                    continue
                norm = self._normalize_token(token)
                parsed = self._try_parse_date(norm)
                if parsed:
                    found_indices.append(idx)

        return found_indices[-1] if found_indices else -1

    def _normalize_token(self, t: str) -> str:
        out = []
        for ch in t:
            out.append(self.OCR_MAP.get(ch, ch))
        s = ''.join(out)
        # collapse repeated separators to single dot, remove leading/trailing dots
        s = re.sub(r'[.]+', '.', s).strip('.')
        return s

    def _try_parse_date(self, s: str, year_min=1900, year_max=2025, default_year: int = None) -> Optional[datetime]:
        """
        Try to parse a date from string `s`.
        Supported input forms (examples):
        - 'dd.mm.yyyy', 'dd.mm.yy'
        - 'ddmmyyyy', 'ddmmyy'
        - 'yyyy.mm.dd'
        - 'dd.mm' or 'dd.m'  (assumes default_year or current year)
        - 'ddmm' (digit-only day+month, assumes default_year or current year)

        Parameters:
        - s: input string
        - year_min/year_max: allowed year range
        - default_year: if provided, use this year when the year is missing;
                        otherwise uses current year.
        Returns:
        datetime on success, otherwise None.
        """
        s = s.strip()
        if not s:
            return None

        # decide default year
        if default_year is None:
            default_year = datetime.now().year
        # clamp default year into allowed bounds
        if default_year < year_min:
            default_year = year_min
        if default_year > year_max:
            default_year = year_max

        # all digits case
        if s.isdigit():
            L = len(s)
            if L == 8:   # ddmmyyyy
                try:
                    d = datetime.strptime(s, '%d%m%Y')
                    if year_min <= d.year <= year_max:
                        return d
                except Exception:
                    pass
            if L == 6:   # ddmmyy
                try:
                    d = datetime.strptime(s, '%d%m%y')
                    yy = int(s[-2:])
                    year = 1900 + yy if yy >= 50 else 2000 + yy
                    d = datetime(year, d.month, d.day)
                    if year_min <= d.year <= year_max:
                        return d
                except Exception:
                    pass
            if L == 4:   # ddmm (no year) -> use default_year
                try:
                    day = int(s[:2])
                    month = int(s[2:])
                    if 1 <= day <= 31 and 1 <= month <= 12:
                        yr = default_year
                        if year_min <= yr <= year_max:
                            return datetime(yr, month, day)
                except Exception:
                    pass
            return None

        # replace any non-digit by dot and split into parts
        parts = re.sub(r'[^0-9]', '.', s).split('.')
        parts = [p for p in parts if p != '']

        # if two parts like "10.05" or "10.5": treat as day.month with default year
        if len(parts) == 2:
            da, mo = parts
            try:
                day = int(da)
                month = int(mo)
                if 1 <= day <= 31 and 1 <= month <= 12:
                    yr_full = default_year
                    if year_min <= yr_full <= year_max:
                        return datetime(yr_full, month, day)
                    else:
                        return None
            except Exception:
                return None

        # expect exactly 3 parts for the other supported dotted forms
        if len(parts) != 3:
            return None

        a, b, c = parts
        # try D-M-Y and Y-M-D variants (day-first default), then M-D-Y
        trials = [
            (a, b, c),  # D M Y
            (c, b, a),  # Y M D
            (b, a, c),  # M D Y (fallback)
        ]
        for da, mo, yr in trials:
            try:
                # two-digit year handling
                if len(yr) == 2:
                    yy = int(yr)
                    yr_full = 1900 + yy if yy >= 50 else 2000 + yy
                else:
                    yr_full = int(yr)
                day = int(da)
                month = int(mo)
                if not (1 <= day <= 31 and 1 <= month <= 12):
                    continue
                if not (year_min <= yr_full <= year_max):
                    continue
                return datetime(yr_full, month, day)
            except Exception:
                continue

        return None