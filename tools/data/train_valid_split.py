import os
import shutil
import natsort
import random
from tqdm.auto import tqdm
from glob import glob

random_seed = 42
random.seed(random_seed)

# 경로 설정
BASE_FOLDER = "/mnt/c/Users/김동영/Downloads/AIFactory/datasets/full"

folder_path = BASE_FOLDER
train_folder = os.path.join(folder_path, 'train')
valid_folder = os.path.join(folder_path, 'valid')
# test_folder = os.path.join(folder_path, 'test')
train_images_folder = os.path.join(train_folder, 'images')
train_labels_folder = os.path.join(train_folder, 'labels')
valid_images_folder = os.path.join(valid_folder, 'images')
valid_labels_folder = os.path.join(valid_folder, 'labels')
# test_images_folder = os.path.join(test_folder, 'images')
# test_labels_folder = os.path.join(test_folder, 'labels')

# train 폴더 생성
os.makedirs(train_folder, exist_ok=True)
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)

# valid 폴더 생성
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(valid_images_folder, exist_ok=True)
os.makedirs(valid_labels_folder, exist_ok=True)

# # test 폴더 생성
# os.makedirs(test_folder, exist_ok=True)
# os.makedirs(test_images_folder, exist_ok=True)
# os.makedirs(test_labels_folder, exist_ok=True)

# 폴더 내 파일 이동
image_file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.png')]

# 리스트를 랜덤하게 섞음
random.shuffle(image_file_list)

train_ratio = 0.8
train_file_count = int(len(image_file_list) * train_ratio)
# valid_file_count = int(len(image_file_list) * 0.9)

for index, file_name in tqdm(enumerate(image_file_list)):
    image_path = os.path.join(folder_path, file_name)
    label_name = os.path.splitext(file_name)[0] + '.txt'
    label_path = os.path.join(folder_path, label_name)
    # json_name = os.path.splitext(file_name)[0] + '.json'
    # json_path = os.path.join(folder_path, json_name)

    if index < train_file_count:
        shutil.move(image_path, os.path.join(train_images_folder, file_name))
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(train_labels_folder, label_name))
        # shutil.move(json_path, os.path.join(train_labels_folder, json_name))
    else:
        shutil.move(image_path, os.path.join(valid_images_folder, file_name))
        if os.path.exists(label_path):
            shutil.move(label_path, os.path.join(valid_labels_folder, label_name))
        # shutil.move(json_path, os.path.join(valid_labels_folder, json_name))  
    # elif index < valid_file_count:
    #     shutil.move(image_path, os.path.join(valid_images_folder, file_name))
    #     shutil.move(label_path, os.path.join(valid_labels_folder, label_name))
    #     # shutil.move(json_path, os.path.join(valid_labels_folder, json_name))
    # else:
    #     shutil.move(image_path, os.path.join(test_images_folder, file_name))
    #     shutil.move(label_path, os.path.join(test_labels_folder, label_name))
    #     # shutil.move(json_path, os.path.join(valid_labels_folder, json_name))
        
print(f"Train : {len(glob('/mnt/c/Users/김동영/Downloads/AIFactory/dataset/train/images/*'))} files moved")
print(f"Valid : {len(glob('/mnt/c/Users/김동영/Downloads/AIFactory/dataset/valid/images/*'))} files moved")
# print(f"Test : {len(glob('/home/steven6774/hdd/Soluray/dataset/test/images/*'))} files moved")
