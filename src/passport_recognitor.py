from src.image_preprocessor import ImagePreprocessor
from typing import List, Tuple, Optional, Iterable
from mrz.checker.td3 import TD3CodeChecker
from rapidfuzz import process, fuzz
from typing import List, Optional
from paddleocr import PaddleOCR
from datetime import datetime
import numpy as np
import math
import cv2
import re


class PassportRecognitor:
    ocr: PaddleOCR

    def __init__(self):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang='ru')
        
        self.image_preprocessor = ImagePreprocessor()
        
        self.foreign_passport_keywords = [
            "FEDERATION","PASSPORT","Surname","Nationality","Place of birth",
            "Type","Code of issuing State","Passport No.","Date of expiry","Date of issue",
            "Sex","Authority","Holder's signature","Given names"
        ]
        self.foreign_passport_keywords = [t.lower() for t in self.foreign_passport_keywords]

        self.foreign_passport_authority_keywords = ["ФМС", "ГУВМ", "МВД", "МИД", "МИД РФ"]

        self.authority_threshold = 80
        
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

        self.LATIN_TO_CYR = {
            'A':'А','B':'В','C':'С','E':'Е','H':'Н','K':'К','M':'М','O':'О','P':'Р','T':'Т','Y':'У','X':'Х',
            'a':'а','b':'в','c':'с','e':'е','o':'о','p':'р','y':'у','x':'х','m':'м','k':'к','t':'т'
        }
        self.EXTRA_VARIANTS = {
            'Φ':'Ф',  # Greek Capital Phi U+03A6
            'φ':'ф',  # Greek Small Phi U+03C6
            'Ѳ':'Ф',  # Cyrillic Fita U+0472
            'ѳ':'ф',  # Cyrillic small fita U+0473
            # при желании можно добавить 'F'->'Ф' и 'f'->'ф' (если OCR даёт латинский F)
            # 'F':'Ф', 'f':'ф',
        }

        all_map = {**self.LATIN_TO_CYR, **self.EXTRA_VARIANTS}
        self.translate_table = str.maketrans({ord(k): ord(v) for k, v in all_map.items()})
        
    def extract_passport_data(self, image: np.ndarray):
        paddle_result = self.ocr.predict(image)

        cropped_image, related_bboxes = self.image_preprocessor.rotate_and_crop(image, paddle_result)

        for bbox in related_bboxes:
            cv2.rectangle(cropped_image, bbox[0].astype(int), bbox[1].astype(int), (0, 0, 255), 3)
            cv2.circle(cropped_image, bbox[0].astype(int), 3, (255, 0, 0))
            cv2.circle(cropped_image, bbox[1].astype(int), 3, (0, 255, 0))
        cv2.imwrite("output.png", cropped_image)


        for res in paddle_result:
            res.print()
            res.save_to_img("output")
            res.save_to_json("output")

        passport_type = self.identify_passport_type(paddle_result[0]["rec_texts"])

        passport_data = {}

        if passport_type == "internal":
            passport_data = self.extract_internal_passport_data(paddle_result, related_bboxes, cropped_image)
        else:
            # passport_data = "Can't extract data from foreign passports yet."
            passport_data = self.extract_foreign_passport_data(paddle_result, related_bboxes, cropped_image)

        return passport_data
    
    def extract_internal_passport_data(self, paddle_result: dict, related_bboxes: list, image):
        mrz_first_line_id = next(
            (idx for idx, text in enumerate(paddle_result[0]["rec_texts"]) if text.endswith("<<<")),
            None  # fallback if not found
        )

        if mrz_first_line_id == None:
            return {"error": "Couldn't find first MRZ line"}

        mrz_second_line_id = next(
            (
                idx
                for idx, text in enumerate(paddle_result[0]["rec_texts"])
                if idx != mrz_first_line_id and "<<<" in text
            ),
            None  # fallback if not found
        )

        if mrz_second_line_id == None:
            return {"error": "Couldn't find second MRZ line"}

        mrz_first_line = paddle_result[0]["rec_texts"][mrz_first_line_id] # len = 44
        mrz_second_line = paddle_result[0]["rec_texts"][mrz_second_line_id] # len = 44

        mrz_first_line_id_box = related_bboxes[mrz_first_line_id]

        try:
            td3_check = TD3CodeChecker(mrz_first_line + "\n" + mrz_second_line)
        except Exception as e:
            return {"error": "MRZ parsing failed"}

        fields = td3_check.fields()

        department_code = fields.optional_data[7:10] + "-" + fields.optional_data[10:13]

        best_depratment_code_match = process.extractOne(department_code, paddle_result[0]["rec_texts"], scorer=fuzz.ratio)
        depratment_code_id = best_depratment_code_match[2]
        depratment_code_bbox = related_bboxes[depratment_code_id]

        birth_date_id = self.find_date_indices(paddle_result[0]["rec_texts"])[-1]

        birth_place_fist_line_id = self.find_closest_bbox_below(birth_date_id, related_bboxes)

        birth_place_bbox_ids = self.find_bboxes_below_between(birth_place_fist_line_id, 
                                                    related_bboxes, 
                                                    mrz_first_line_id_box[0][1] - 10)

        birth_place_bbox_ids = [birth_place_fist_line_id] + birth_place_bbox_ids
        
        birth_place_lines = [paddle_result[0]["rec_texts"][i] for i in birth_place_bbox_ids]
        birth_place = " ".join(birth_place_lines)

        best_russian_match = process.extractOne("РОССИЙСКАЯ", paddle_result[0]["rec_texts"], scorer=fuzz.ratio)
        russian_score = best_russian_match[1]
        russian_id = best_russian_match[2]

        best_federation_match = process.extractOne("ФЕДЕРАЦИЯ", paddle_result[0]["rec_texts"], scorer=fuzz.ratio)
        federation_score = best_federation_match[1]
        federation_id = best_federation_match[2]

        best_russian_federation_match = process.extractOne("РОССИЙСКАЯ ФЕДЕРАЦИЯ", paddle_result[0]["rec_texts"], scorer=fuzz.ratio)
        russian_federation_score = best_russian_federation_match[1]
        russian_federation_id = best_russian_federation_match[2]

        rus_fed_scores = [russian_score, federation_score, russian_federation_score]

        max_score_id = np.argmax(rus_fed_scores)

        chosen_id = russian_id
        authority_search_start_point = np.array(related_bboxes[russian_id][1]).astype(int)
        if max_score_id == 1:
            fed_bbox = related_bboxes[federation_id]
            authority_search_start_point = np.array([fed_bbox[0][0], fed_bbox[1][1]]).astype(int)
            chosen_id = federation_id
        elif max_score_id == 2:
            rus_fed_bbox = related_bboxes[russian_federation_id]
            mean_x = (rus_fed_bbox[0][0] + rus_fed_bbox[1][0]) / 2
            authority_search_start_point = np.array([mean_x, rus_fed_bbox[1][1]]).astype(int)
            chosen_id = russian_federation_id
            

        authority_first_line_id = self.find_bboxes_below_point(authority_search_start_point, 
                                            related_bboxes,
                                            ignore_indices=[chosen_id])[0]
        
        authority_bbox_ids = self.find_bboxes_below_between(authority_first_line_id, 
                                                            related_bboxes, 
                                                            depratment_code_bbox[0][1] - 10)
        
        authority_bbox_ids = [authority_first_line_id] + authority_bbox_ids


        authority_lines = [paddle_result[0]["rec_texts"][i] for i in authority_bbox_ids]
        authority = " ".join(authority_lines)

        output_dict: dict = {}

        surname = self.mrz_name_to_cyrillic(fields.surname)
        name_and_patronymic = fields.name.split(" ")

        output_dict["passport_surname"] = surname
        output_dict["passport_name"] = self.mrz_name_to_cyrillic(name_and_patronymic[0])
        output_dict["passport_patronymic"] = self.mrz_name_to_cyrillic(name_and_patronymic[1]) if len(name_and_patronymic) > 1 else ""

        output_dict["passport_sex"] = fields.sex # Пол
        
        birth_year = fields.birth_date[0:2]
        birth_month = fields.birth_date[2:4]
        birth_day = fields.birth_date[4:6]

        output_dict["passport_birth_date"] = f"{birth_day}.{birth_month}.{birth_year}"

        output_dict["passport_birth_place"] = birth_place

        output_dict["passport_series"] = fields.document_number[0:3] + fields.optional_data[0]
        output_dict["passport_number"] = fields.document_number[3:]

        issue_year = fields.optional_data[1:3]
        issue_month = fields.optional_data[3:5]
        issue_day = fields.optional_data[5:7]


        output_dict["passport_issue_date"] = f"{issue_day}.{issue_month}.{issue_year}"

        output_dict["passport_department_code"] = department_code

        output_dict["passport_authority"] = authority

        return output_dict
    
    def find_closest_bbox_below(
            self,
            source_bbox_id: int,
            bboxes: List[List[List[float]]],
            min_overlap_frac: float = 0.2,
            epsilon: float = 1e-6,
            y_increases_down: bool = True,
        ) -> int:
            """
            Find the index of the closest bounding box that lies strictly below the source bbox
            and has horizontal overlap with it. Returns -1 if none found.

            Assumes bbox format: [[x_min, y_min], [x_max, y_max]]
            and that coordinates are in the same space for all boxes.

            Parameters
            ----------
            min_overlap_frac :
                Minimum fraction of horizontal overlap relative to source width required to
                consider a candidate (default 0.2 = 20% overlap).
            epsilon :
                Small tolerance to avoid numeric edge cases.
            y_increases_down :
                True if y coordinates increase downward (typical image coordinates).
                If False, the vertical comparisons are inverted.
            """
            if not (0 <= source_bbox_id < len(bboxes)):
                raise IndexError("source_bbox_id out of range")

            sx_min, sy_min = bboxes[source_bbox_id][0]
            sx_max, sy_max = bboxes[source_bbox_id][1]
            s_center_x = (sx_min + sx_max) / 2.0
            s_width = max(epsilon, sx_max - sx_min)

            closest_id = -1
            min_vertical_gap = float("inf")
            best_overlap = -1.0
            best_center_dist = float("inf")

            for i, bbox in enumerate(bboxes):
                if i == source_bbox_id:
                    continue

                cx_min, cy_min = bbox[0]
                cx_max, cy_max = bbox[1]

                # Normalize ordering (in case somebody passed inverted coordinates)
                if cx_min > cx_max:
                    cx_min, cx_max = cx_max, cx_min
                if cy_min > cy_max:
                    cy_min, cy_max = cy_max, cy_min

                # vertical_gap definition depends on y direction
                if y_increases_down:
                    # candidate top (cy_min) minus source bottom (sy_max)
                    vertical_gap = cy_min - sy_max
                else:
                    # if y increases up, then candidate bottom should be less than source top
                    # so compute positive gap if candidate is below in that coordinate system
                    vertical_gap = sy_min - cy_max

                # We want strictly below: require vertical_gap > epsilon
                if vertical_gap <= epsilon:
                    # candidate starts at or above the source bottom -> not strictly below
                    continue

                # horizontal overlap length and fraction relative to source width
                overlap_len = max(0.0, min(sx_max, cx_max) - max(sx_min, cx_min))
                overlap_frac = overlap_len / s_width

                if overlap_len <= 0 or overlap_frac < min_overlap_frac:
                    # not enough horizontal overlap
                    continue

                center_dist = abs(((cx_min + cx_max) / 2.0) - s_center_x)

                # prefer smaller vertical gap, then larger overlap, then closer center
                better = False
                if vertical_gap < min_vertical_gap - 1e-9:
                    better = True
                elif abs(vertical_gap - min_vertical_gap) < 1e-9:
                    if overlap_len > best_overlap + 1e-9:
                        better = True
                    elif abs(overlap_len - best_overlap) < 1e-9 and center_dist < best_center_dist - 1e-9:
                        better = True

                if better:
                    closest_id = i
                    min_vertical_gap = vertical_gap
                    best_overlap = overlap_len
                    best_center_dist = center_dist

            return closest_id

    def find_bboxes_below_between(
        self,
        source_bbox_id: int,
        bboxes: List[List[List[float]]],
        y_border: Optional[float] = None,
        require_horizontal_overlap: bool = True,
        include_touching: bool = True
    ) -> List[int]:
        """
        Return indices of bboxes that are entirely below the source bbox (not inside or above).
        Assumes bbox format: bbox = [[x_min, y_min], [x_max, y_max]] (top-left at index 0, bottom-right at index 1).
        include_touching=True allows candidate top == source bottom (touching); otherwise strict >.
        """
        if not (0 <= source_bbox_id < len(bboxes)):
            raise IndexError("source_bbox_id out of range")

        sx_min, sy_min = bboxes[source_bbox_id][0]
        sx_max, sy_max = bboxes[source_bbox_id][1]
        s_center_x = (sx_min + sx_max) / 2.0

        eps = 1e-9
        results = []

        for i, bbox in enumerate(bboxes):
            if i == source_bbox_id:
                continue

            cx_min, cy_min = bbox[0]   # candidate top (y_min)
            cx_max, cy_max = bbox[1]   # candidate bottom (y_max)

            # --- must be entirely below source ---
            if include_touching:
                # allow touching: candidate top may equal source bottom
                if cy_min + eps < sy_max:
                    # candidate's top is above source bottom -> inside/overlapping/above -> skip
                    continue
            else:
                # require strictly below
                if cy_min - eps <= sy_max:
                    continue

            # if y_border provided: keep only candidates above that border
            # (i.e., candidate top <= y_border under top-left origin)
            if y_border is not None:
                if include_touching:
                    if cy_min > y_border:
                        continue
                else:
                    if cy_min >= y_border:
                        continue

            # horizontal overlap check
            overlap = min(sx_max, cx_max) - max(sx_min, cx_min)
            if require_horizontal_overlap and overlap <= 0:
                continue

            vertical_gap = cy_min - sy_max  # >= 0 when below (0 if touching allowed)
            center_dist = abs(((cx_min + cx_max) / 2.0) - s_center_x)

            results.append((i, vertical_gap, overlap, center_dist))

        # sort by: smallest vertical gap, then largest overlap, then smallest center distance
        results.sort(key=lambda t: (t[1], -t[2], t[3]))

        return [t[0] for t in results]




    
    def find_bboxes_below_point(
        self,
        point: Tuple[float, float],
        bboxes: List[List[List[float]]],
        horizontal_tolerance: float = 0.0,
        include_touching: bool = True,
        max_distance: Optional[float] = None,
        limit: Optional[int] = None,
        ignore_indices: Optional[Iterable[int]] = None
    ) -> List[int]:
        px, py = point
        ignore_set = set(ignore_indices) if ignore_indices is not None else set()
        results = []
        eps = 1e-9

        for i, bbox in enumerate(bboxes):
            if i in ignore_set:
                continue

            x_min, y_min = bbox[0]
            x_max, y_max = bbox[1]

            # raw vertical difference between candidate top and the point
            vertical_gap_raw = y_min - py

            # Decide acceptance + compute vertical_gap (non-negative)
            if vertical_gap_raw > eps:
                # candidate top is strictly below the point -> OK
                vertical_gap = vertical_gap_raw
            elif abs(vertical_gap_raw) <= eps:
                # candidate top == point y (touching at top)
                if not include_touching:
                    continue
                vertical_gap = 0.0
            else:
                # vertical_gap_raw < 0: candidate top is above the point
                # it may still *contain* the point (y_min < py <= y_max)
                # -> treat as gap = 0 (accepted). Otherwise it's above the point -> skip.
                if py <= y_max + eps:
                    # point is inside the bbox vertically (or on bottom edge) -> accept with gap 0
                    vertical_gap = 0.0
                else:
                    # point is below the bbox entirely -> exclude
                    continue

            # horizontal distance: 0 if point is inside [x_min, x_max], otherwise distance to nearest edge
            if x_min <= px <= x_max:
                horiz_dist = 0.0
            elif px < x_min:
                horiz_dist = x_min - px
            else:  # px > x_max
                horiz_dist = px - x_max

            if horiz_dist > horizontal_tolerance:
                continue

            if (max_distance is not None) and (vertical_gap > max_distance):
                continue

            results.append((i, vertical_gap, horiz_dist))

        # sort by vertical gap (closest first) then horizontal distance
        results.sort(key=lambda t: (t[1], t[2]))

        indices = [t[0] for t in results]
        if limit is not None:
            return indices[:limit]
        return indices

    def find_closest_in_dir(self, origin_pos: list, origin_id: int, dir: list, polyes: list, source_line_length: int = 100) -> int:
        pt1 = origin_pos
        pt2 = (int(pt1[0] + dir[0] * source_line_length), 
            int(pt1[1] + dir[1] * source_line_length))

        source_line = [pt1, pt2]

        found_idx = 0
        min_dist = 0
        for idx, poly in enumerate(polyes):
            if idx == origin_id:
                continue

            current_line = [poly[0], poly[1]]

            dist = self.lines_intersect_distance(source_line, current_line)
            if dist is not None:
                if found_idx == 0 or dist < min_dist:
                    found_idx = idx
                    min_dist = dist

        return found_idx
    
    def extract_foreign_passport_data(self, paddle_result: dict, related_bboxes: list, image):
        # mrz_first_line = paddle_result[0]["rec_texts"][-2]
        # mrz_second_line = paddle_result[0]["rec_texts"][-1]
        mrz_first_line = next((text for text in paddle_result[0]["rec_texts"] if text.endswith("<<<")), None)
        mrz_second_line = next((text for text in paddle_result[0]["rec_texts"] if text != mrz_first_line and "<<<" in text), None)

        td3_check = TD3CodeChecker(mrz_first_line + "\n" + mrz_second_line)

        date_indices = self.find_date_indices(paddle_result[0]["rec_texts"])

        fields = td3_check.fields()

        output_dict: dict = {}

        best_surname_match = process.extractOne(fields.surname, paddle_result[0]["rec_texts"], scorer=fuzz.ratio)
        lat_surname_id = best_surname_match[2]
        lat_surname = paddle_result[0]["rec_texts"][lat_surname_id]
        cyr_surname = paddle_result[0]["rec_texts"][lat_surname_id - 1]

        best_name_match = process.extractOne(fields.name, paddle_result[0]["rec_texts"], scorer=fuzz.ratio)
        lat_name_id = best_name_match[2]
        lat_name = paddle_result[0]["rec_texts"][lat_name_id]
        cyr_name = paddle_result[0]["rec_texts"][lat_name_id - 1]

        cyr_patronymic = ""
        rus_name_splitted = cyr_name.split()
        if len(rus_name_splitted) > 1:
            cyr_name_splitted = cyr_name.split()
            cyr_name = cyr_name_splitted[0]
            cyr_patronymic = cyr_name_splitted[-1]
        
        best_id = -1
        best_score = -1
        for keyword in self.foreign_passport_authority_keywords:
            for rec_text_id, rec_text in enumerate(paddle_result[0]["rec_texts"]):
                rec_text_splitted = rec_text.split()
                if len(rec_text_splitted) > 1:
                    rec_text = rec_text.rsplit(' ', 1)[0]
                
                score = fuzz.ratio(self.latin_to_cyr(rec_text), keyword)

                if score > best_score:
                    best_id = rec_text_id
                    best_score = score
        
        authority = ""
        if best_id >= 0 and best_score >= self.authority_threshold:
            authority = paddle_result[0]["rec_texts"][best_id]

        output_dict["foreign_authority"] = authority

        output_dict["foreign_lat_surname"] = lat_surname
        output_dict["foreign_lat_name"] = lat_name

        output_dict["foreign_cyr_surname"] = cyr_surname
        output_dict["foreign_cyr_name"] = cyr_name
        output_dict["foreign_patronymic"] = cyr_patronymic

        output_dict["foreign_sex"] = fields.sex
        
        birth_year = fields.birth_date[0:2]
        birth_month = fields.birth_date[2:4]
        birth_day = fields.birth_date[4:6]

        output_dict["foreign_birth_date"] = f"{birth_day}.{birth_month}.{birth_year}"

        output_dict["foreign_nationality"] = fields.nationality

        output_dict["foreign_issue_date"] = paddle_result[0]["rec_texts"][date_indices[-2]]

        expiry_year = fields.expiry_date[0:2]
        expiry_month = fields.expiry_date[2:4]
        expiry_day = fields.expiry_date[4:6]

        output_dict["foreign_expiry_date"] = f"{expiry_day}.{expiry_month}.{expiry_year}"

        output_dict["foreign_document_number"] = fields.document_number

        output_dict["mrz_1"] = mrz_first_line
        output_dict["mrz_2"] = mrz_second_line

        return output_dict

    def latin_to_cyr(self, s: str) -> str:
        """Заменяет латинские буквы из таблицы на кириллические.
        Остальные символы остаются без изменений."""
        return s.translate(self.translate_table)
    
    
    # def count_scripts(self, s: str):
    #     cyr = 0
    #     lat = 0
    #     for ch in s:
    #         # простая проверка по диапазонам Unicode
    #         if '\u0400' <= ch <= '\u04FF' or '\u0500' <= ch <= '\u052F':
    #             cyr += 1
    #         elif 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
    #             lat += 1
    #     return cyr, lat

    # def normalize_homoglyphs(self, s: str) -> str:
    #     """Преобразует символы из меньшинственного скрипта в доминирующий по внешнему виду."""
    #     cyr, lat = self.count_scripts(s)
    #     # если кириллицы больше или равно — приводим латиницу к кириллице
    #     if cyr >= lat:
    #         mapping = self.LATIN_TO_CYR
    #     else:
    #         mapping = self.CYR_TO_LATIN
    #     return ''.join(mapping.get(ch, ch) for ch in s)

    
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


    def find_date_indices(self, lines: List[str]) -> int:
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

        return found_indices

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