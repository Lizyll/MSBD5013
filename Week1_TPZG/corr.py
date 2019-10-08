
import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


data = pd.read_csv('test_data2.csv', header = 0)
df = data.corr(method='spearman')

df.to_csv('spearcorr.csv', encoding='utf-8')