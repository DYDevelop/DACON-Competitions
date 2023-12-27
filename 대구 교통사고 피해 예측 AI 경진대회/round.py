import numpy as np
import pandas as pd

original_csv = pd.read_csv('rmsle_original/submission.csv')
original_csv = original_csv.round(1)
original_csv.to_csv('rounded.csv', index=False)

print('Done!')
