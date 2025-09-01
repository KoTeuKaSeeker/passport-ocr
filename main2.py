from src.passport_recognitor import PassportRecognitor
from PIL import Image
import numpy as np

if __name__ == "__main__":
    passport_recognitor = PassportRecognitor()
    # image = np.array(Image.open("data/2312660261.jpg"))
    image = np.array(Image.open("data/646128.w450.jpg"))
    

    passport_data = passport_recognitor.extract_passport_data(image)

    print(passport_data)