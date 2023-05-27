# DACON 대회 코드
## 차례
1. 월간 데이콘 항공편 지연 예측 AI 경진대회
- 알고리즘 | 정형 | 분류 | 준지도학습 | 항공 | LogLoss
-  2023.04.03 ~ 2023.05.08 09:59
-  959명

2. 도배 하자 유형 분류 AI 경진대회
- 알고리즘 | 비전 | 분류 | MLOps | Weighted F1 Score
- 2023.04.10 ~ 2023.05.22 09:59
- 1,478명 

3. 합성데이터 기반 객체 탐지 AI 경진대회
- 알고리즘 | 비전 | 객체 탐지 | mAP
- 2023.05.08 ~ 2023.06.19 09:59
- 218명


## mmdetection 2.28.2 - Training with Custom Dataset

1-1) mmdetection > mmdet > datasets > trash.py (TrashDataset class) 를 새로 생성한다.

* 내용은 coco.py를 복붙하되, 클래스 명과 CLASSES변수값만 커스텀하도록 한다. (COCODataset -> CarDataset, __init__에 추가해주기)

 
 
1-2) mmdetection > configs > _base_ > datasets > trash_detection.py 를 새로 생성한다

* 내용은 coco_detection.py를 복붙하되, dataset_type과 data_root dict값을 변경한다. (데이터 경로, COCODataset -> CarDataset)

 
 
1-3) mmdetection > configs > faster_rcnn > faster_rcnn_r50_fpn_1x_trash.py를 새로 생성한다.

* 내용은 faster_rcnn_r50_fpn_1x_coco.py를 복붙하되, '../_base_/datasets/coco_detection.py',를 변경한다. (bbox_head=dict(num_classes=??,) 등등 변경)



1-4) 기타 Config file 수정한다.

* _base_/default_runtime.py안에 auto_scale_lr = dict(enable=True, base_batch_size=16)로 변경, --seed 설정, --deterministic 설정 등 Hyperparameter 설정


# 1. 월간 데이콘 항공편 지연 예측 AI 경진대회

준지도학습을 통한 항공편 지연 예측 ML 모델 개발 

이 대회를 통해 다양한 M.L 모델들과 여러개의 AUTOML을 사용해보는 경험을 쌓을 수 있었다.

|Auto M.L|Model|Log_Loss|Things i did|
|---|---|---|---|
|pycaret|Linear regression|0.647| 'Origin_Airport', 'Destination_Airport', 'Cancelled' 제외,  빈도수 각 column 마다 10회 미만 삭제  = 총  445713개 -> 443369개|
|pycaret|Linear regression|0.7387|Baseline Code 최빈값으로 대체,  'Origin_Airport', 'Destination_Airport' 제외, 총 255001개|
|pycaret|Top 3 Stacked|0.6477|Baseline Code 최빈값으로 대체,  'Origin_Airport', 'Destination_Airport' 제외, 총 255001개|
|pycaret|Top 3 Bagged|0.6477|Baseline Code 최빈값으로 대체,  'Origin_Airport', 'Destination_Airport' 제외, 총 255001개|
|x|Linear regression|0.6477|날짜 및 시간 수치를 Sin 함수에 넣어 반복되는 형태로 변형|
|H20|Linear regression|0.6660|날짜 및 시간 수치를 Sin 함수에 넣어 반복되는 형태로 변형|
|H20|Linear regression|0.6487|날짜 및 시간 수치를 Sin 함수에 넣어 반복되는 형태로 변형, Binary Class Under sampling|
|TPOT|Linear regression|2.8214|날짜 및 시간 수치를 Sin 함수에 넣어 반복되는 형태로 변형, Binary Class Under sampling|
|H20|Linear regression|1.0544|날짜 및 시간 수치를 Sin 함수에 넣어 반복되는 형태로 변형, Binary Class Over sampling|
|H20|Gradient boosting|0.6509|날짜 및 시간 수치를 Sin 함수에 넣어 반복되는 형태로 변형, Binary Class Under sampling|
|FLAML|catboost|0.9080|날짜 및 시간 수치를 Sin 함수에 넣어 반복되는 형태로 변형|
|H20|Linear regression|0.6982|날짜 및 시간 수치를 Sin 함수에 넣어 반복되는 형태로 변형, Binary Class Under sampling, 결과 소숫점 버림|
|H20|Linear regression|0.6796|날짜 및 시간 수치를 Sin 함수에 넣어 반복되는 형태로 변형, Binary Class Under sampling, 결과 첫번째 소숫점 버림|

# 2. 도배 하자 유형 분류 AI 

한솔데코는 끊임없는 도전을 통해 성장을 모색하고자 하는 기치를 갖고, 공동 주택 내 실내 마감재 공사를 수행하며 시트와 마루, 벽면, 도배 등 건축에서 빼놓을 수 없는 핵심적인 자재를 유통하고 있습니다.  

실내 마감재는 건축물 내부 공간의 인테리어와 쾌적한 생활을 좌우하는 만큼, 제품 결함에 대한 꼼꼼한 관리 역시 매우 중요합니다.  

이를 위해 한솔데코에서는 AI 기술을 활용하여 하자를 판단하고 빠르게 대처할 수 있는 혁신적인 방안을 모색하고자 합니다.  

이 대회를 통해 직접 Object Detection을 위한 BBox Labeling, 디양한 CNN 모델과 ViT(Vision Transformer)들을 사용해보는 좋은 기회가 되었다. 
