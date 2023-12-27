import json

def coco_to_labelme(coco_file_path, output_dir):
    # COCO 형식의 어노테이션 파일 불러오기
    with open(coco_file_path, 'r') as f:
        coco_data = json.load(f)
    
    # Labelme 형식의 JSON을 담을 리스트 초기화
    labelme_annotations = []

    # COCO 데이터를 Labelme 형식으로 변환
    for image_info in coco_data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        # Labelme의 각 이미지 정보를 초기화
        labelme_image = {
            "version": "4.5.6",
            "flags": {},
            "shapes": [],
            "imagePath": file_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        # COCO 어노테이션에서 해당 이미지의 레이블과 바운딩 박스 정보 추출
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                category_id = annotation['category_id']
                category_name = coco_data['categories'][category_id - 1]['name']

                # 바운딩 박스 좌표 추출
                bbox = annotation['bbox']
                x, y, w, h = bbox
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h

                # Labelme 형식의 shape 정보 생성
                shape_info = {
                    "label": category_name,
                    "points": [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }

                labelme_image['shapes'].append(shape_info)

        labelme_annotations.append(labelme_image)

    # Labelme 형식의 JSON 파일로 저장
    for i, labelme_image in enumerate(labelme_annotations):
        name = labelme_image['imagePath'].replace('.png', '.json')
        output_file_path = f'{output_dir}/{name}'
        with open(output_file_path, 'w') as f:
            json.dump(labelme_image, f)

if __name__ == '__main__':
    coco_file_path = 'base5.json'
    output_dir = 'datasets/test'

    coco_to_labelme(coco_file_path, output_dir)
