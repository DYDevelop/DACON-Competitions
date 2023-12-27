import os
import shutil
import natsort
import random
from tqdm.auto import tqdm
from glob import glob

random_seed = 42
random.seed(random_seed)

# 경로 설정
folder_path = "/mnt/c/Users/김동영/Downloads/AIFactory/datasets/train/imagesempty/train"
move_images_folder = '/mnt/c/Users/김동영/Downloads/AIFactory/datasets/empty/train'

# 폴더 내 파일 이동
image_file_list = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.png')]

empty_image_list = []

for file_name in tqdm(enumerate(image_file_list)):
    label_name = file_name[1].replace('png', 'txt')
    label_path = os.path.join(folder_path.replace('images', 'labels'), label_name)
    if os.path.isfile(label_path): continue
    else: empty_image_list.append(file_name[1])

print(len(empty_image_list))
# 리스트를 랜덤하게 섞음
random.shuffle(empty_image_list)

# random.shuffle(image_file_list)

move_ratio = 1.0
# idx = 4429
train_file_count = int(len(empty_image_list) * move_ratio)

for index, image_name in tqdm(enumerate(empty_image_list)):
    image_path = os.path.join(folder_path, image_name)
    if index < move_ratio:
        shutil.move(image_path, os.path.join(move_images_folder, image_name))
    else: break
