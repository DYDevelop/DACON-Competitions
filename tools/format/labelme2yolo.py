from glob import glob
from tqdm.auto import tqdm
import json as JSON

def labelme_to_yolo(labelme_file_path, output_file_path, class_mapping):
    with open(labelme_file_path, 'r') as f:
        labelme_data = JSON.load(f)

    image_width = labelme_data['imageWidth']
    image_height = labelme_data['imageHeight']

    for shape in labelme_data['shapes']:
        label = shape['label']
        points = shape['points']

        # 중심 좌표 계산
        x_center = (points[0][0] + points[1][0]) / 2
        y_center = (points[0][1] + points[2][1]) / 2

        # 너비와 높이 계산
        width = abs(points[1][0] - points[0][0])
        height = abs(points[2][1] - points[0][1])

        # YOLO 형식의 텍스트로 변환
        class_idx = class_mapping[label]
        yolo_format = f"{class_idx} {x_center/image_width} {y_center/image_height} {width/image_width} {height/image_height}"

        # 파일에 쓰기
        with open(output_file_path, 'a') as yolo_file:
            yolo_file.write(yolo_format + '\n')

if __name__ == '__main__':
    json_list = glob('datasets/test/*.json')
    
    for json in tqdm(json_list):
        labelme_file_path = json
        output_file_path = json.replace('json', 'txt')
        class_mapping = {'농어':0, '베스':1, '숭어':2, '강준치':3, '블루길':4, '잉어':5, '붕어':6, '누치':7}  # 클래스 이름과 YOLO 클래스 번호 매핑

        labelme_to_yolo(labelme_file_path, output_file_path, class_mapping)
