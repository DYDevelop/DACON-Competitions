import numpy as np
import numpy.typing as npt
import cv2, os, shutil
from glob import glob
from tqdm.auto import tqdm

def image_bounding_box_crop(image: npt.NDArray[np.uint8], x, y, w, h):
    bounding_cropped_image = image[
        y : y + h,
        x : x + w,
    ]
    return bounding_cropped_image

def product_image_rect_contour(image: npt.NDArray[np.uint8]):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contour_index = 0
    boxes = []

    for contur in contours:
        x, y, w, h = cv2.boundingRect(contur)
        if w > 100 and h > 100:
            boxes.append([x, y, w, h])

    # Sort contours by ratio in descending order
    boxes = sorted(boxes, key=lambda x:x[2] / x[3] if x[2] < x[3] else x[3] / x[2], reverse=True)
    if not boxes:
        return 0, 0, 0, 0
    product_contour = boxes[contour_index]

    return product_contour

img_paths = sorted(glob('/Users/KDY/Downloads/real/*'))
if os.path.exists('/Users/KDY/Downloads/cropped'):
    shutil.rmtree('/Users/KDY/Downloads/cropped')
os.makedirs('/Users/KDY/Downloads/cropped', exist_ok=True)
cnt = 0
for img_path in tqdm(img_paths):
    x, y, w, h = product_image_rect_contour(cv2.imread(img_path))
    if h != 0 and w != 1000:
        img_crop = image_bounding_box_crop(cv2.imread(img_path), x, y, w, h)
        cv2.imwrite(f"./cropped/{img_path.split('/')[-1]}", img_crop)
        cnt += 1
print(f"{cnt} were saved!")