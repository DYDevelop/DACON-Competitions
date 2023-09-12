import json
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

ann_path = '/home/steven6774/hdd/mmdetection-2.28.2/custom_dataset/aug_dataset/*.txt'
ann_files = glob(ann_path)

# valid set을 나눌 경우
ann_train, ann_valid = train_test_split(ann_files,
                                       test_size=0.2,
                                       random_state=1119)

def labelmetxt2coco(anns):
    pork = {}
    # 숫자로 해도 되지만, 클래스 확인하기 위해 차종으로 변경
    classes = ["needle", "dot"]
    pork["categories"] = [{"id":i,"name":cat,"supercategory":"none"} for i, cat in enumerate(classes)]
    pork["images"] = []
    pork["annotations"] = []
    cnt_ann = 0
    for i, ann in enumerate(anns):
        img_path = ann.replace("txt", "bmp")
        h, w, _ = cv2.imread(img_path).shape
        pork["images"].append({"id":i,"height":h,"width":w,"file_name":img_path})

        f_ann = open(ann, "r")
        for line in f_ann.readlines():
            data = line.split()
            cat = int(float(data[0]))
            x = float(data[1]) * w
            y = float(data[2]) * h
            width = float(data[3]) * w
            height = float(data[4]) * h
            # pt3x = int(data[5])
            # pt3y = int(data[6])
            # #pt4x = int(data[7])
            # #pt4y = int(data[8])
            # x = pt1x
            # y = pt1y
            # width = pt3x - pt1x
            # height = pt3y - pt1y
            area = width * height
            pork["annotations"].append({"id": cnt_ann,
                                        "image_id": i,
                                        "category_id": cat,
                                        "bbox": [x, y, width, height],
                                        "area": area,
                                        "segmentation": [],
                                        "iscrowd": 0})
            cnt_ann += 1
        
    return pork

# with open('/content/drive/MyDrive/DACON/236107/data/annotations/train_json', "w") as f:
#     json.dump(labelmetxt2coco(ann_files), f, ensure_ascii=False, indent=4)

# train/valid split을 했을 경우
with open('/home/steven6774/hdd/mmdetection-2.28.2/custom_dataset/aug_dataset/annotations/train_json.json', "w") as f:
    json.dump(labelmetxt2coco(ann_train), f, ensure_ascii=False, indent=4)

with open('/home/steven6774/hdd/mmdetection-2.28.2/custom_dataset/aug_dataset/annotations/val_json.json', "w") as f:
    json.dump(labelmetxt2coco(ann_valid), f, ensure_ascii=False, indent=4)

print(f'Total of {len(ann_files)} images and labels dumped!')