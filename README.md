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
- 1,463명


## mmdetection v2.28.2 - Training with Custom Dataset

1-1) mmdetection > mmdet > datasets > trash.py (TrashDataset class) 를 새로 생성한다.

* 내용은 coco.py를 복붙하되, 클래스 명과 CLASSES변수값만 커스텀하도록 한다. (COCODataset -> CarDataset, __init__에 추가해주기)

 
 
1-2) mmdetection > configs > _base_ > datasets > trash_detection.py 를 새로 생성한다

* 내용은 coco_detection.py를 복붙하되, dataset_type과 data_root dict값을 변경한다. (데이터 경로, COCODataset -> CarDataset)

 
 
1-3) mmdetection > configs > faster_rcnn > faster_rcnn_r50_fpn_1x_trash.py를 새로 생성한다.

* 내용은 faster_rcnn_r50_fpn_1x_coco.py를 복붙하되, '../_base_/datasets/coco_detection.py',를 변경한다. (bbox_head=dict(num_classes=??,) 등등 변경)



1-4) 기타 Config file 수정한다.

* _base_/default_runtime.py안에 auto_scale_lr = dict(enable=True, base_batch_size=16)로 변경, --seed 설정, --deterministic 설정 등 Hyperparameter 설정<br>

1-5) Start Training<br>
```shell
# 단일 GPU로 학습
python tools/train.py configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py --seed 42

# 2개 GPU로 학습
bash tools/dist_train.sh configs/sparse_rcnn/sparse_rcnn_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py 2
```


Info) Custom Module이 regitery에 등록 안될때

* mmcv-full 및 mmdet을 재설치 하거나 밑의 코드를 실행해 mmdet 외 requerments를 설치
```shell
# install mmcv-full
pip install -U openmim
mim install mmcv-full

# install mmdet
cd mmdetection
pip install -v -e.
```

* Batch size를 늘리고 싶지만 memory가 부족할때 config 파일에 추가하기 (Auto Mixed Precision)
```shell
# fp16 settings
fp16 = dict(loss_scale=512.)
```
Info) 버전 문제가 많아 사용한 버전들을 적어두자
- mmdetection v2.28.2 with python=3.8
```shell
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
mim mmcv-full==1.7.0
```
- mmdetection v3.2.0 with python=3.8 (WSL2 on Windows)
```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
Info) Could not load library libcudnn_cnn_infer.so.8. Error: libcuda.so: cannot open shared object file: No such file or directory<br>
```shell
sudo apt update
sudo apt install nvidia-cudnn
```


# 1. 월간 데이콘 항공편 지연 예측 AI 경진대회

준지도학습을 통한 항공편 지연 예측 ML 모델 개발 (정형데이터 대회 처음)

이 대회를 통해 다양한 M.L 모델들과 여러개의 AUTOML을 사용해보는 경험을 쌓을 수 있었다.

|Auto M.L|Model|Log_Loss|Things i did|
|:---:|:---:|:---:|:-------------------------:|
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

이 대회를 통해 직접 Object Detection을 위한 BBox Labeling, 디양한 CNN 모델과 ViT(Vision Transformer)들을 사용해보는 좋은 기회가 되었다. (비정형 데이터 대회 처음)   

- 가장먼저 어떤 모델이 가장 높은 Val Weighted F1 Score가 나오는지 확인 


|Model|Epochs|Train Loss|Val Loss|Val Weighted F1 Score|Parms|
|:---:|:---:|:---:|:---:|:---:|:---:|
|efficientnet_b0|10|0.02804|1.13945|0.78437|5.3M|
|efficientnet_b7|10|0.09705|1.84711|0.72508|66M|
|efficientnet_v2_s|10|0.07720|1.12502|0.78861|22M|
|efficientnet_v2_m|10|0.15804|1.0013|0.79867|120M|
|efficientnet_v2_l|30|0.00178|0.87607|0.83241|120M|
|maxvit_t|10|0.01741|1.16789|0.78507|69M|
|maxvit_t|30|0.00658|0.8173|0.84318|69M|
|swin_v2_b|10|0.11221|1.13066|0.75830|88M| 

- maxvit_t모델이 학습 시간대비 가장 높은 Val Weighted F1 Score가 나와 이 모델을 선택함 


|Model|Folds|CutMix|RandomContrast|HorizontalFlip|MixUp|Weighted F1 Score|Things i did|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:--------------------------------------------------------------------:|
|efficientnet_b7|10|X|O|O|O|0.58577||
|efficientnet_b7|10|X|X|X|X|0.58844||
|maxvit_t|10|X|O|O|O|0.61362||
|maxvit_t|10|O|O|O|X|0.62905||
|maxvit_t_Stacked|10|O|O|O|X|0.62190|분류를 잘 못하는 3 Class 분류기 Stacked|
|maxvit_t|10|O|O|O|O|0.61120||
|maxvit_t|10|O|O|O|X|0.60196|Focal Loss로 변경|
|maxvit_t|10|O|O|O|X|0.61659|Early Stop 10|
|maxvit_t|10|X|O|O|O|0.61096|부족한 Class에 대해 데이터 크롤링하여 추가함|
|maxvit_t|10|X|X|X|X|0.60418||
|maxvit_t|10|O|O|O|X|0.62086|CenterCrop(p=0.5, height=300, width=300)|
|maxvit_t|10|O|X|O|X|0.61445|AdamW, ReduceLROnPlateau, Data Added|
|maxvit_t|10|O|X|O|X|0.63913|AdamW, CosineLR, Data Relabeled(일부만)|
|maxvit_t|5|O|X|O|X|0.58788|AdamW, CosineLR, Data Relabeled -full(전체)|
|maxvit_t|10|O|X|O|X|0.61463|AdamW, CosineLR, Data Relabeled -full(전체)| 

- Object Detection Methods

|Model|Epoch|Weighted F1 Score|Things i did|
|:---:|:---:|:---:|:---:|
|YOLOv4|20|0.5751|Pre-Trained|
|YOLOv6|200|0.5220|Pre-Trained|

# 3. 합성데이터 기반 객체 탐지 AI 경진대회
  
합성데이터란 실제 환경에서 수집되거나 측정되는 것이 아니라 디지털 환경에서 생성되는 데이터셋으로,

최근 방대한 양질의 데이터셋이 필요해짐에 따라 그 중요성이 대두되고 있습니다.

합성 데이터는 데이터 라벨링 작업을 위한 2배 이상의 시간 절약과 10배 가까운 비용을 절감하게 하고, 자동화를 바탕으로 정확한 라벨링의 데이터 그리고 정확한 AI 모델 개발을 위한 데이터의 다양화를 가능하게 합니다.

본 경진대회는 실사와 같은 객체와 배경을 비솔의 3D Rendering과 VFX 기술로 생성된 AI 학습용 고품질 합성데이터를 바탕으로 진행됩니다.

- 다양한 데이터 강화 기법 사용
- 모델 앙상블을 통한 일반화
- 다양한 모델의 성능 비교
- Two-step으로 나누어 Detection 후 분류 시도 (Over Fitting)

|Model|Epoch|mAP|Things i did|
|:---:|:---:|:---:|:---:|
|Faster R-CNN|12|0.731|-|
|Faster R-CNN|12|0.735|이상박스 탐지 후 제거|
|Faster R-CNN|12|0.896|Mixup|
|Faster R-CNN|12|0.900|Mixup, 이상박스 탐지 후 제거|
|Faster R-CNN|12|0.921|Mixup, 이상박스 탐지 후 제거|
|Grid R-CNN|12|0.889|Mixup|
|Faster R-CNN|24|0.928|Mixup, Photo Distortion, 이상박스 탐지 후 제거|
|Libra R-CNN|24|0.954|Mixup, Photo Distortion, 이상박스 탐지 후 제거|
|Libra R-CNN|24|0.931|Mixup, Photo Distortion, 이상박스 탐지 후 제거, Backbone=ResNeXt152|
|Libra R-CNN|24|0.955|Mixup, Photo Distortion, 추가 Detecor로 이상박스 탐지 후 제거|
|Libra R-CNN|24|0.957|Mixup, Photo Distortion, 추가 Detecor로 이상박스 탐지 후 제거, AutoAugment|
|Cascade R-CNN|24|0.937|Mixup, Photo Distortion, 추가 Detecor로 이상박스 탐지 후 제거, AutoAugment|
|Cascade R-CNN|20|0.941|Mixup, Photo Distortion, 추가 Detecor로 이상박스 탐지 후 제거, AutoAugment|

- CSV Ensembled (Model Ensemble)
   
|Models|mAP|
|:---:|:---:|
|Libra + Cascade|0.979|
|Libra + Cascade + Postprocessing|0.981|
