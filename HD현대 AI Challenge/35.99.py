import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import bisect
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import re
import optuna
from optuna.integration import XGBoostPruningCallback
sns.set_theme(style="darkgrid")

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv').drop(columns=['SAMPLE_ID'])
test = pd.read_csv('test.csv').drop(columns=['SAMPLE_ID'])

# datetime 컬럼 처리
train['ATA'] = pd.to_datetime(train['ATA'])
test['ATA'] = pd.to_datetime(test['ATA'])

# train.drop(columns=['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN'], inplace=True)
# test.drop(columns=['U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN'], inplace=True)

# datetime을 여러 파생 변수로 변환
for df in tqdm([train, test]):
    df['year'] = df['ATA'].dt.year
    df['month'] = df['ATA'].dt.month
    df['day'] = df['ATA'].dt.day
    df['hour'] = df['ATA'].dt.hour
    df['minute'] = df['ATA'].dt.minute
    df['weekday'] = df['ATA'].dt.weekday
    # df['Time_of_Day'] = np.where(df['hour'] < 12, 'AM', 'PM')
    # df['All_minute'] = df['hour'] * 60 + df['minute']
    # df['Date'] = df['month'].astype(str) + '-' + df['day'].astype(str)
    # df['Dest_Location'] = df['ARI_CO'] + '/' + df['ARI_PO']
    # df['Month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    # df['Month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # df['Prev_month'] = np.where(df['month'] == 1, 12, df['month'] - 1)
    # df['Diff_local'] = np.abs(df['hour'] - df['ATA_LT'])
    # df['Square'] = df['LENGTH'] * df['BREADTH'] * df['DEPTH']
    # # df['Width'] = df['PORT_SIZE'] / df['DIST']
    # df['Is_Full'] = df['GT'] / df['DEADWEIGHT'] * 100
    # # df['Wind'] = (df['U_WIND'] ** 2 + df['V_WIND'] ** 2) ** 0.5

# datetime 컬럼 제거
train.drop(columns='ATA', inplace=True)
test.drop(columns='ATA', inplace=True)

# Categorical 컬럼 인코딩
categorical_features = train.select_dtypes(include=['object']).columns.tolist()
encoders = {}

for feature in tqdm(categorical_features, desc="Encoding features"):
    le = LabelEncoder()
    train[feature] = le.fit_transform(train[feature].astype(str))
    le_classes_set = set(le.classes_)
    test[feature] = test[feature].map(lambda s: '-1' if s not in le_classes_set else s)
    le_classes = le.classes_.tolist()
    bisect.insort_left(le_classes, '-1')
    le.classes_ = np.array(le_classes)
    test[feature] = le.transform(test[feature].astype(str))
    encoders[feature] = le

# 결측치 처리
train.fillna(train.mean(), inplace=True)
test.fillna(train.mean(), inplace=True)

train_X = train.drop(columns='CI_HOUR')
train_y = train['CI_HOUR']

h_train_X, h_valid_X, h_train_y, h_valid_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

# SEED 50이 가장 좋았음

lgbm_params = {'random_state': 42,
                # "device_type": 'gpu',
                'n_estimators': 45000,
                'objective' : 'mae',
                'num_leaves': 31,
                'min_child_samples': 12,
                'learning_rate': 0.17010396907527026,
                'colsample_bytree': 0.9605563464803123,
                'reg_alpha': 0.1110993344544235,
                'reg_lambda': 0.7948637803974561,
                'verbose': -1}

final_lgb_model = lgb.LGBMRegressor(**lgbm_params)
final_lgb_model.fit(train_X, train_y)
final_lgb_pred = final_lgb_model.predict(test)
print(mean_absolute_error(h_valid_y, final_lgb_model.predict(h_valid_X)))

submit = pd.read_csv('sample_submission.csv')
submit['CI_HOUR'] = final_lgb_pred
submit.to_csv('baseline_submit.csv', index=False)
