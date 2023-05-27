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

준지도학습은 데이터의 레이블링에 드는 비용을 절약할 수 있어 다양한 사업 분야에서 활발히 쓰이고 있습니다. 

이번 월간 데이콘 항공편 지연 예측 AI 경진대회에서는 바로 이러한 준지도학습을 활용한 AI 모델 개발을 목표로 합니다. 

무궁무진한 가능성을 품고 있는 준지도학습에 대한 이해도를 높여 보세요! 
