import json
import os
import natsort
import json
import cv2

# 텍스트 읽기
def read_text(file_path, encoding="utf-8"):
    text = None
    with open(file_path, 'r', encoding=encoding) as f:
        text = f.read()

    return text
def read_lines(file_path, encoding="utf-8"):
    return read_text(file_path, encoding).split("\n")

def convert_yolo_to_labelme(yolo_file, json_file, image_file, image_path):
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_file,
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    with open(yolo_file, "r") as file:
        for line in file:
            label, x, y, width, height = line.strip().split(" ")
            x, y, width, height = map(float, [x, y, width, height])
            x1 = image_width*x - (image_width*width)/2
            y1 = image_height*y - (image_height*height)/2
            x2 = image_width*x + (image_width*width)/2
            y2 = image_height*y + (image_height*height)/2

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > image_width:
                x2 = image_width
            if y2 > image_height:
                y2 = image_height

            shape = {
                "label": all_class_dict[label],
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }

            labelme_data["shapes"].append(shape)

    with open(json_file, "w") as file:
        # json.dump(labelme_data, file)
        json.dump(labelme_data, file, indent=4)

        print("saved at: ", json_file)

if __name__ == "__main__":
    # class 라벨 가져오기
    CLASS_LABEL = r"classes.txt"

    class_list = read_lines(CLASS_LABEL)
    all_class_dict = {}
    for class_index, class_text in enumerate(class_list):
        all_class_dict[str(class_index)] = class_text

    print(all_class_dict)

    # 이미지 폴더 경로
    LABEL_FOLDER = r"/home/steven6774/hdd/Soluray/닭/test/labels"
    IMAGE_FOLDER = r"/home/steven6774/hdd/Soluray/닭/test/images"
    file_num = len(os.listdir(LABEL_FOLDER))
    for i, file in enumerate(natsort.natsorted(os.listdir(LABEL_FOLDER))):
        if file != ".DS_Store":
            if file.endswith(".txt"):
                # 원본 txt파일이 있는 폴더 경로
                text_path = os.path.join(LABEL_FOLDER, file)

                json_file = os.path.splitext(file)[0] + ".json"
                json_path = os.path.join(LABEL_FOLDER, json_file)
                image_file = os.path.splitext(file)[0] + ".bmp"
                image_path = os.path.join(IMAGE_FOLDER, image_file)

                convert_yolo_to_labelme(text_path, json_path, image_file, image_path)
                # print(file, "\n", f"{i}/{file_num}", end="\r", flush=True)