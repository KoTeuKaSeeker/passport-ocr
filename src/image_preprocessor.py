import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import os
import random
import colorsys

class ImagePreprocessor:
    def __init__(self):
        self.GOLDEN_RATIO_CONJ = 0.618033988749895

    def rotate_and_crop(self, image, paddlel_results):
        

        # result = ocr.predict(image_rgb)
        rects = self.get_rects(paddlel_results)
        
        mid_angle = np.median([r[2] for r in rects])
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2.0, h/2.0), mid_angle, 1.0).astype(np.float64)
        
        all_rotated = self.get_rotated_bboxes(rects, M) 
        rotated_corners = self.calculate_rotated_bbox_corners(all_rotated)
        crop_points = [self.inverse_rotate_point(corner, M) for corner in rotated_corners]

        related_bboxes = self.get_related_boxes(all_rotated, rotated_corners[0])

        # IT SHOULDN'T BE THIS WAY! DON'T FORGET TO FIX IT!
        for idx, bbox in enumerate(related_bboxes):
            related_bboxes[idx][0][1], related_bboxes[idx][1][1] = related_bboxes[idx][1][1], related_bboxes[idx][0][1]


        cropped = self.crop_rotated_rect_perspective(image, crop_points)

        return cropped, related_bboxes
    
    def get_positional_curves(self, norm_bboxes):
        curves = []
        for box in norm_bboxes:
            (x1, y1), (x2, y2) = box

            x_curve = []
            y_curve = []
            for other_box in norm_bboxes:
                (ox1, oy1), (ox2, oy2) = other_box

                right_side = ox1 - x2
                left_side = x1 - ox2
                x_dist = max(left_side, right_side)
                x_curve.append(x_dist)

                top_side = y1 - oy2
                bottom_side = oy1 - y2
                y_dist = max(top_side, bottom_side)
                y_curve.append(y_dist)
            
            x_curve, y_curve = sorted(x_curve), sorted(y_curve)
            curves.append((x_curve, y_curve))
        return curves

    def inverse_rotate_point(self, pt, M):
        R = M[:, :2]   # 2x2
        t = M[:, 2]    # 2,
        R_inv = np.linalg.inv(R)
        pt_before = R_inv.dot(pt - t)
        return pt_before

    def get_related_boxes(self, all_rotated, left_top_corner):
        related_bboxes = []
        for box_points in all_rotated:
            box = box_points[[0, 2]] - left_top_corner
            related_bboxes.append(box)
        return related_bboxes
    
    def calculate_rotated_bbox_corners(self, all_rotated):
        all_rotated_v = np.vstack(all_rotated)  # (4*num_rects, 2)

        min_x = all_rotated_v[:, 0].min()
        min_y = all_rotated_v[:, 1].min()
        max_x = all_rotated_v[:, 0].max()
        max_y = all_rotated_v[:, 1].max()


        rotated_p0 = np.array([min_x, min_y], dtype=np.float64)
        rotated_p1 = np.array([max_x, min_y], dtype=np.float64)
        rotated_p2 = np.array([max_x, max_y], dtype=np.float64)
        rotated_p3 = np.array([min_x, max_y], dtype=np.float64)

        return np.array([rotated_p0, rotated_p1, rotated_p2, rotated_p3])

    def get_rotated_bboxes(self, rects, M):
        all_rotated = []  # соберём все повёрнутые точки
        for rect in rects:
            box = cv2.boxPoints(rect).astype(np.float64)  # (4,2)
            # cv2.transform ожидает shape (1, N, 2) или (N,1,2) — так надёжно:
            rotated = cv2.transform(box[None, :, :], M)[0]  # (4,2)
            all_rotated.append(rotated)
        
        return all_rotated
    
    def canonical_rect(self, rect):
        """
        Вход: rect_or_pts может быть:
        - RotatedRect ( (cx,cy),(w,h),angle )   -> то, что возвращает cv2.minAreaRect
        - ndarray shape (4,2)                   -> 4 вершины (boxPoints)
        - ndarray shape (N,2)                   -> контур/полигон (тогда внутри вызовем minAreaRect)
        Возврат: canonical_rot ( (cx,cy), (w,h), angle_deg ) где w >= h и angle_deg — угол ДЛИННОЙ стороны относительно оси X в градусах,
                angle в диапазоне (-90, 90].
        """
        # попытка получить 4 вершины коробки
        arr = np.asarray(rect)

        if arr.ndim == 2 and arr.shape[1] == 2:
            if arr.shape[0] == 4:
                box = arr.astype(np.float64)
            else:
                # это контур/полигон — найдём минимальный area rect и boxPoints
                rect = cv2.minAreaRect(arr.astype(np.float32))
                box = cv2.boxPoints(rect).astype(np.float64)
        else:
            # возможно передали RotatedRect (tuple/list length 3)
            try:
                box = cv2.boxPoints(rect).astype(np.float64)
            except Exception as e:
                raise ValueError("Не удалось распознать вход: ожидается RotatedRect или ndarray Nx2") from e

        # берём два соседних ребра, чтобы сравнить длины
        v0 = box[1] - box[0]
        v1 = box[2] - box[1]
        l0 = np.hypot(v0[0], v0[1])
        l1 = np.hypot(v1[0], v1[1])

        if l0 >= l1:
            long_vec = v0
            long_len = l0
            short_len = l1
        else:
            long_vec = v1
            long_len = l1
            short_len = l0

        # угол длинной стороны относительно оси X (deg). atan2: положительный = против часовой
        angle = np.degrees(np.arctan2(long_vec[1], long_vec[0]))

        # нормализуем в (-90, 90]
        if angle <= -90:
            angle += 180
        elif angle > 90:
            angle -= 180

        center = box.mean(axis=0)
        w = float(long_len)
        h = float(short_len)

        return ( (float(center[0]), float(center[1])), (w, h), float(angle) )

    def get_rects(self, paddle_ocr_result):
        rects = []
        for polygon in paddle_ocr_result[0]["dt_polys"]:
            rect = self.canonical_rect(polygon)
            rects.append(rect)
        return rects

    def crop_rotated_rect_perspective(self, img, box_points):
        """
        box_points: ndarray (4,2) — четыре вершины прямоугольника в порядке, как возвращает cv2.boxPoints
        Возвращает: вырез (h, w, channels)
        """
        box = np.asarray(box_points, dtype=np.float32)

        # вычислим ширину и высоту (берём максимум между противоположными ребрами)
        widthA = np.linalg.norm(box[1] - box[0])
        widthB = np.linalg.norm(box[2] - box[3])
        maxWidth = max(int(round(widthA)), int(round(widthB)))

        heightA = np.linalg.norm(box[2] - box[1])
        heightB = np.linalg.norm(box[3] - box[0])
        maxHeight = max(int(round(heightA)), int(round(heightB)))

        # Корректный порядок: (tl, tr, br, bl). Если boxPoints даёт другой порядок,
        # убедись, что ты передаёшь правильно — обычно cv2.boxPoints даёт корректный порядок.
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(box, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
        return warped
    
    def _hsv_to_rgb255(self, h, s, v):
        """h,s,v in [0,1] -> (r,g,b) in 0..255 ints"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(round(r*255)), int(round(g*255)), int(round(b*255)))

    def rgb255_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def random_beautiful_color(self, 
                               seed=None,
                               hue=None,
                               sat_range=(0.55, 0.95),
                               val_range=(0.65, 0.95)):
        """
        Возвращает (hex, (r,g,b)) — цвет в HEX и RGB(0-255).
        Если hue задан (0..1), используется он; иначе берём случайный.
        sat_range, val_range — диапазоны для насыщенности и яркости.
        """
        if seed is not None:
            random.seed(seed)
        h = hue if hue is not None else random.random()
        s = random.uniform(*sat_range)
        v = random.uniform(*val_range)
        rgb = self._hsv_to_rgb255(h % 1.0, s, v)
        return self.rgb255_to_hex(rgb), rgb

    def generate_palette(self, n, seed=None, method="golden", sat_range=(0.55,0.95), val_range=(0.65,0.95)):
        """
        Генерирует n разных «красивых» цветов (список HEX).
        method: "uniform" (равномерно по hue) или "golden" (золотое сечение для хорошего распределения).
        """
        if seed is not None:
            random.seed(seed)
        palette = []
        if method == "uniform":
            for i in range(n):
                h = i / n
                s = random.uniform(*sat_range)
                v = random.uniform(*val_range)
                rgb = self._hsv_to_rgb255(h, s, v)
                palette.append(rgb)
        else:  # golden
            h = random.random()
            for i in range(n):
                h = (h + self.GOLDEN_RATIO_CONJ) % 1.0
                s = random.uniform(*sat_range)
                v = random.uniform(*val_range)
                rgb = self._hsv_to_rgb255(h, s, v)
                palette.append(rgb)
        return palette

if __name__ == "__main__":
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='ru')
    
    image = cv2.imread("data/dataset/24.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = ocr.predict(image_rgb)

    image_preprocessor = ImagePreprocessor()

    cropped_image, related_bboxes = image_preprocessor.rotate_and_crop(image, result)

    for bbox in related_bboxes:
        cv2.rectangle(cropped_image, bbox[0].astype(int), bbox[1].astype(int), (0, 0, 255), 3)
    
    cv2.imwrite("cropped_image.png", cropped_image)
