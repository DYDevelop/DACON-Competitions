import pandas as pd
import numpy as np
import random
import os
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')
    
csv_to_parquet('./train.csv', 'train')
csv_to_parquet('./test.csv', 'test')

train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')
sample_submission = pd.read_csv('sample_submission.csv', index_col = 0)

#레이블(Delay)을 제외한 결측값이 존재하는 변수들을 학습 데이터의 최빈값으로 대체합니다
NaN_col = ['Origin_State','Destination_State','Airline','Estimated_Departure_Time', 'Estimated_Arrival_Time','Carrier_Code(IATA)','Carrier_ID(DOT)']

for col in NaN_col:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    
    if col in test.columns:
        test[col] = test[col].fillna(mode)
print('Done.')

#레이블이 없는 데이터들을 제거합니다
train = train.dropna()

column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
    
def to_number(x, dic):
    return dic[x]

train.loc[:, 'Delay_num'] = train['Delay'].apply(lambda x: to_number(x, column_number))
print('Done.')

train_x = train.drop(columns=['ID', 'Delay', 'Delay_num'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])

from pycaret.classification import *
from sklearn.metrics import log_loss

exp_name = setup(data=train_x, target=train_y, use_gpu=True)
add_metric('logloss', 'Log Loss', log_loss, greater_is_better = False)
best_model = compare_models(fold=5, sort = 'logloss', 
                                    include=['lr', 'ridge', 'lda', 'dummy', 'knn', 'qda'])

print(best_model) # Log Loss : 6.0951

best_tune = tune_model(best_model, n_iter=50, choose_better = True)
evaluate_model(best_tune)

y_pred = predict_model(best_tune, data= test_x)[['prediction_label']]
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)
submission.to_csv('baseline_submission.csv', index=True)