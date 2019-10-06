import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


# BestAvgTable 

BestAvgTable = pd.read_csv('BestAvgTable.csv', index_col='class')

# input file
rid = 'r8'
filename = rid + '.csv'
column = ['horse', 'distance', 'es_finishtime']
#column = ['horse', 'distance', 'es_finishtime', 'first', 'firstthree']
matchofday = {'r1': ('G3', 1000), 'r2': ('G3', 1400), 'r3': ('Class 2', 1200), 
	'r4': ('Class 3', 1200), 'r5': ('Class 3', 1400), 'r6': ('Class 3', 1800),
	'r7': ('Class 4', 1200), 'r8': ('Class 4', 1200), 'r9': ('Class 4', 1400), 
	'r10': ('Class 4', 1600), }
r = pd.read_csv(filename, usecols=column, squeeze=True)
r = r.dropna(how='all')
print(r)


# find the Best_avg value
val6 = BestAvgTable.at[matchofday[rid][0], str(matchofday[rid][1])]


list_Pred = list(zip(r.horse, r.es_finishtime))

#print(list_Pred)
predict = list()
for (x, y) in list_Pred:
	t = (x, y)
	p = (val6 - y) / val6
	t = t + (p ,)
	predict.append(t)
	
predict.sort(key=lambda tup: tup[2], reverse=True)
first = predict[0]
first3 = predict[:3]

res1 = list()
name = list()
res2 = list()

for (x, y) in list_Pred:
	if first[0] == x:
		res1.append(1)
	else:
		res1.append(0)

for (x, y, z) in first3:
	name.append(x)

for (x, y) in list_Pred:
	if x in name:
		res2.append(1)
	else:
		res2.append(0)

print('The predicted result of', rid, 'is as below:')
print('The first place prediction: ', res1)
print('The first three prediction: ', res2)

r['first'] = res1
r['firstthree'] = res2

r.to_csv(filename, encoding='utf-8')