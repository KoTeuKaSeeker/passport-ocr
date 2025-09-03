import os
from src.passport_recognitor import PassportRecognitor
from rapidfuzz import fuzz
from PIL import Image
import numpy as np
import logging

sasha_data = {
    'passport_surname': 'ДОЛГОВ', 
    'passport_name': 'АЛЕКСАНДР', 
    'passport_patronymic': 'ПЕТРОВИЧ', 
    'passport_sex': 'M', 
    'passport_birth_date': '10.05.03', 
    'passport_birth_place': 'Г. УЛЬЯНОВСК', 
    'passport_series': '7323', 
    'passport_number': '542628', 
    'passport_issue_date': '03.07.23', 
    'passport_department_code': '730-001', 
    'passport_authority': 'УМВД РОССИИ ПО УЛЬЯНОВСКОЙ ОБЛАСТИ'
}

def compare_to_ground_truth(prediction: dict, ground_truth: dict, threshold: float = 80) -> bool:
    for key, gt_value in ground_truth.items():
        if key not in prediction:
            return False
        score = fuzz.ratio(gt_value, prediction[key])

        if score < threshold:
            return False
    return True


if __name__ == "__main__":
    sasha_images_path = "data/sasha_simple"
    sasha_image_names = os.listdir(sasha_images_path)

    passport_recognitor = PassportRecognitor()

    accuracy = 0
    for idx, image_name in enumerate(sasha_image_names):
        image_path = os.path.join(sasha_images_path, image_name)
        image = np.array(Image.open(image_path))
        passport_data = passport_recognitor.extract_passport_data(image)

        passed = False
        if "error" in passport_data:
            print(f"{image_path}: {passport_data['error']}")
        else:
            passed = compare_to_ground_truth(passport_data, sasha_data)
        
        accuracy += 1 if passed else 0

        print(f"{accuracy}/{idx + 1}")
    
    accuracy /= len(sasha_image_names)

    print(f"Total accuracy: {accuracy}")