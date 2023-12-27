import os
import shutil
import natsort
import random
from tqdm.auto import tqdm
from glob import glob

random_seed = 42
random.seed(random_seed)

# 경로 설정
BASE_FOLDER = "/home/steven6774/hdd/mmdetection-2.28.0/custom_dataset/aug_dataset"

train_folder = os.path.join(BASE_FOLDER, 'train')
valid_folder = os.path.join(BASE_FOLDER, 'valid')

# train 폴더 생성
os.makedirs(train_folder, exist_ok=True)

# valid 폴더 생성
os.makedirs(valid_folder, exist_ok=True)

# 폴더 내 파일 이동
image_file_list = glob(BASE_FOLDER + '/*.bmp')
# print(image_file_list)

# 리스트를 랜덤하게 섞음
random.shuffle(image_file_list)

train_ratio = 0.8
train_file_count = int(len(image_file_list) * train_ratio)

for index, file_name in enumerate(image_file_list):
    image_path = file_name
    label_name = file_name.replace('.bmp', '.txt')
    label_path = label_name
    # json_name = os.path.splitext(file_name)[0] + '.json'
    # json_path = os.path.join(folder_path, json_name)
    # print(image_path)
    # print(label_path)


    if index < train_file_count:
        shutil.move(image_path, os.path.join(train_folder, image_path.split('/')[-1]))
        shutil.move(label_path, os.path.join(train_folder, label_name.split('/')[-1]))
        # shutil.move(json_path, os.path.join(train_labels_folder, json_name))
        # print("moved to train/images and train/labels")
    else:
        shutil.move(image_path, os.path.join(valid_folder, file_name.split('/')[-1]))
        shutil.move(label_path, os.path.join(valid_folder, label_name.split('/')[-1]))
        # shutil.move(json_path, os.path.join(valid_labels_folder, json_name))
        # print("moved to valid/images and valid/labels")
