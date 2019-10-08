from __future__ import division
import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame



def t2s(t):
	t = str(t)
	if t != '0':
		m, s, ms = t.strip().split('.')
		ts = float(m) * 60 + float(s) + float(ms) * (1/100)
		return float(round(ts, 2))
	else:
		return 0
'''
def class2n(s):
	if s != 'Hong Kong Group Three':
		s1, s2 = s.strip().split(' ')
		return int(s2)
	else:
		return 0
'''
# load file
data = pd.read_csv('test_data.csv', header = 0).fillna(0)
#print(data)

# count jockey unique value
print(data.jname.value_counts())
# count trainer unique value
print(data.tname.value_counts())


# convert min to s
l = list()
for column in data['besttime']:
	l.append(t2s(column))
data['besttime'] = l
print(data['besttime'])

l1 = list()
#print(data['finishm'])
for column in data['finishm']:
	l1.append(column / 100)
data['finishm'] = l1

cols = ['finishm', 'besttime']
data[cols] = data[cols].replace(0, np.nan)

print(data['besttime'])

data.to_csv('test_data2.csv', encoding='utf-8')
