import os
import random
import numpy as np
import pandas as pd

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

import warnings
warnings.filterwarnings('ignore')

light_df = pd.read_csv('data/external_open/대구 보안등 정보.csv', encoding='cp949')[['설치개수', '소재지지번주소']]

location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

light_df[['도시', '구', '동', '번지']] = light_df['소재지지번주소'].str.extract(location_pattern)
light_df = light_df.drop(columns=['소재지지번주소', '번지'])

light_df = light_df.groupby(['도시', '구', '동']).sum().reset_index()
light_df.reset_index(inplace=True, drop=True)

child_area_df = pd.read_csv('data/external_open/대구 어린이 보호 구역 정보.csv', encoding='cp949').drop_duplicates()[['시설종류', '소재지지번주소', '관할경찰서명', 'CCTV설치여부']]
child_area_df['cnt'] = 1

location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

child_area_df[['도시', '구', '동', '번지']] = child_area_df['소재지지번주소'].str.extract(location_pattern)
child_area_df = child_area_df.drop(columns=['소재지지번주소', '번지'])

child_area_df = child_area_df.groupby(['도시', '구', '동']).sum().reset_index()
child_area_df.reset_index(inplace=True, drop=True)

parking_df = pd.read_csv('data/external_open/대구 주차장 정보.csv', encoding='cp949')[['주차장구분', '주차장유형', '주차구획수', '급지구분', '요금정보', '소재지지번주소']]
parking_df = pd.get_dummies(parking_df, columns=['급지구분'])

location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

parking_df[['도시', '구', '동', '번지']] = parking_df['소재지지번주소'].str.extract(location_pattern)
parking_df = parking_df.drop(columns=['소재지지번주소', '번지'])

parking_df = parking_df.groupby(['도시', '구', '동']).sum().reset_index()
parking_df.reset_index(inplace=True, drop=True)

cctv_df = pd.read_csv('data/external_open/대구 CCTV 정보.csv', encoding='cp949')[['소재지지번주소', '단속구분', '도로노선방향']]
cctv_df = pd.get_dummies(cctv_df, columns=['단속구분', '도로노선방향'])

location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

cctv_df[['도시', '구', '동', '번지']] = cctv_df['소재지지번주소'].str.extract(location_pattern)
cctv_df = cctv_df.drop(columns=['소재지지번주소', '번지'])

cctv_df = cctv_df.groupby(['도시', '구', '동']).sum().reset_index()
cctv_df.reset_index(inplace=True, drop=True)

train_org = pd.read_csv('data/train.csv')
test_org = pd.read_csv('data/test.csv')

train_df = train_org.copy()
test_df = test_org.copy()

time_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2})'

train_df[['연', '월', '일', '시간']] = train_org['사고일시'].str.extract(time_pattern)
train_df[['연', '월', '일', '시간']] = train_df[['연', '월', '일', '시간']].apply(pd.to_numeric) # 추출된 문자열을 수치화해줍니다
train_df = train_df.drop(columns=['사고일시']) # 정보 추출이 완료된 '사고일시' 컬럼은 제거합니다

# 해당 과정을 test_x에 대해서도 반복해줍니다
test_df[['연', '월', '일', '시간']] = test_org['사고일시'].str.extract(time_pattern)
test_df[['연', '월', '일', '시간']] = test_df[['연', '월', '일', '시간']].apply(pd.to_numeric)
test_df = test_df.drop(columns=['사고일시'])

location_pattern = r'(\S+) (\S+) (\S+)'

train_df[['도시', '구', '동']] = train_org['시군구'].str.extract(location_pattern)
train_df = train_df.drop(columns=['시군구'])

test_df[['도시', '구', '동']] = test_org['시군구'].str.extract(location_pattern)
test_df = test_df.drop(columns=['시군구'])

road_pattern = r'(.+) - (.+)'

train_df[['도로형태1', '도로형태2']] = train_org['도로형태'].str.extract(road_pattern)
train_df = train_df.drop(columns=['도로형태'])

test_df[['도로형태1', '도로형태2']] = test_org['도로형태'].str.extract(road_pattern)
test_df = test_df.drop(columns=['도로형태'])

# train_df와 test_df에, light_df와 child_area_df, parking_df를 merge하세요.
train_df = pd.merge(train_df, light_df, how='left', on=['도시', '구', '동'])
train_df = pd.merge(train_df, child_area_df, how='left', on=['도시', '구', '동'])
train_df = pd.merge(train_df, parking_df, how='left', on=['도시', '구', '동'])
train_df = pd.merge(train_df, cctv_df, how='left', on=['도시', '구', '동'])

test_df = pd.merge(test_df, light_df, how='left', on=['도시', '구', '동'])
test_df = pd.merge(test_df, child_area_df, how='left', on=['도시', '구', '동'])
test_df = pd.merge(test_df, parking_df, how='left', on=['도시', '구', '동'])
test_df = pd.merge(test_df, cctv_df, how='left', on=['도시', '구', '동'])

test_x = test_df.drop(columns=['ID', '도시']).copy()
train_x = train_df[test_x.columns].copy()
train_y = train_df['ECLO'].copy()

train_x['주차장구분'] = np.where(train_x['주차장구분'].str.contains('민영'), '민영', '공영')
train_x['주차장유형'] = np.where(train_x['주차장유형'].str.contains('노상'), '노상', '노외')
train_x['요금정보'] = np.where(train_x['요금정보'].str.contains('유로'), '유로', '무료')
train_x['CCTV설치여부'] = np.where(train_x['CCTV설치여부'].str.contains('Y'), 'Y', 'N')

test_x['주차장구분'] = np.where(test_x['주차장구분'].str.contains('민영'), '민영', '공영')
test_x['주차장유형'] = np.where(test_x['주차장유형'].str.contains('노상'), '노상', '노외')
test_x['요금정보'] = np.where(test_x['요금정보'].str.contains('유로'), '유로', '무료')
test_x['CCTV설치여부'] = np.where(test_x['CCTV설치여부'].str.contains('Y'), 'Y', 'N')

# train_x['주차장구분'] = np.where(train_x['주차장구분'].str.contains('민영'), '민영', np.where(train_x['주차장구분'].notna(), '공영', np.nan))
# train_x['주차장유형'] = np.where(train_x['주차장유형'].str.contains('노상'), '노상', np.where(train_x['주차장유형'].notna(), '노외', np.nan))
# train_x['요금정보'] = np.where(train_x['요금정보'].str.contains('유로'), '유로', np.where(train_x['요금정보'].notna(), '무료', np.nan))
# train_x['CCTV설치여부'] = np.where(train_x['CCTV설치여부'].str.contains('Y'), 'Y', np.where(train_x['CCTV설치여부'].notna(), 'N', np.nan))

# test_x['주차장구분'] = np.where(test_x['주차장구분'].str.contains('민영'), '민영', np.where(test_x['주차장구분'].notna(), '공영', np.nan))
# test_x['주차장유형'] = np.where(test_x['주차장유형'].str.contains('노상'), '노상', np.where(test_x['주차장유형'].notna(), '노외', np.nan))
# test_x['요금정보'] = np.where(test_x['요금정보'].str.contains('유로'), '유로', np.where(test_x['요금정보'].notna(), '무료', np.nan))
# test_x['CCTV설치여부'] = np.where(test_x['CCTV설치여부'].str.contains('Y'), 'Y', np.where(test_x['CCTV설치여부'].notna(), 'N', np.nan))

from sklearn.preprocessing import LabelEncoder

categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index)

for i in categorical_features:
    le = LabelEncoder()
    le=le.fit(train_x[i]) 
    train_x[i]=le.transform(train_x[i])
    test_x[i]=le.transform(test_x[i])

# from category_encoders.target_encoder import TargetEncoder

# categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index)

# for i in categorical_features:
#     le = TargetEncoder(cols=[i])
#     train_x[i] = le.fit_transform(train_x[i], train_y)
#     test_x[i] = le.transform(test_x[i])

train_x.fillna(0, inplace=True)
test_x.fillna(0, inplace=True)

X_train, Y_train, test = train_x, train_y, test_x

from supervised.automl import AutoML
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(X_train)
test_scaled = scaler.transform(test)

X_train = pd.DataFrame(train_scaled, columns = X_train.columns)
test = pd.DataFrame(test_scaled, columns = test.columns)

def rmsle(y_true, y_predicted, sample_weight=None):
    log_y = np.log1p(y_true)
    log_pred = np.log1p(y_predicted)
    squared_error = (log_y-log_pred)**2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

automl = AutoML(
    algorithms=["CatBoost", "Xgboost", "LightGBM", "Random Forest"],
    mode="Compete",
    ml_task="regression",
    eval_metric="rmse",
    random_state=42,
    total_time_limit=None,
    model_time_limit=None
)

automl.fit(X_train, Y_train)

sample_submission = pd.read_csv('data/sample_submission.csv')

sample_submission["ECLO"] = automl.predict(test)

sample_submission.to_csv("submission.csv", index=False)