from src.passport_recognitor import PassportRecognitor
from PIL import Image
import numpy as np

if __name__ == "__main__":
    passport_recognitor = PassportRecognitor()

    paths = [
        "data/dataset/6.jpg",
        "data/images/sasha_2.jpg",
        "data/images/danil.jpg",
        "data/dataset/24.jpg",
        "data/images/2.jpg"
        ]
    
    for path in paths:
        print("-" * 5 + path + "-" * 5)
        image = np.array(Image.open(path))
        passport_data = passport_recognitor.extract_passport_data(image)
        print(passport_data)

    # image = np.array(Image.open("data/2312660261.jpg"))
    # image = np.array(Image.open("data/dataset/6.jpg"))
    # image = np.array(Image.open("data/images/sasha.jpg"))
    # image = np.array(Image.open("data/images/sasha_2.jpg"))
    # image = np.array(Image.open("data/images/danil.jpg"))
    # image = np.array(Image.open("data/dataset/24.jpg"))
    # image = np.array(Image.open("data/images/2.jpg"))
    

    # passport_data = passport_recognitor.extract_passport_data(image)

    # print(passport_data)