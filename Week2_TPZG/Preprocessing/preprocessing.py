import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# cal joc trainer
data_ori = pd.read_csv('test_data3.csv')
# all jockey and trainer
jockeydict = data_ori.jname.value_counts().to_dict()
trainerdict = data_ori.tname.value_counts().to_dict()
# all place 3 jockey and trainer
plapool = data_ori[data_ori['ind_pla'] == 1]
jockeywindict = plapool.jname.value_counts().to_dict()
trainerwindict = plapool.tname.value_counts().to_dict()

# get dict: trainwin40, jocwin40
jocwin = dict()
for key, val in jockeydict.items():
	if key in jockeywindict:
		value = jockeywindict.get(key) / val
		jocwin.update({key: value})

trainwin = dict()
for key, val in trainerdict.items():
	if key in trainerwindict:
		value = trainerwindict.get(key) / val
		trainwin.update({key: value})
'''
joc = [ v for k, v in jocwin.items()]
joc40 = np.percentile(joc, 40, axis=0)
jocwin40 = dict()
for k, v in jocwin.items():
	if v >= joc40:
		jocwin40.update({k: 1})
	else:
		jocwin40.update({k : 0})

tra = [ v for k, v in trainwin.items()]
tra40 = np.percentile(tra, 40, axis=0)
trainwin40 = dict()
for k, v in trainwin.items():
	if v >= tra40:
		trainwin40.update({k: 1})
	else:
		trainwin40.update({k: 0})
'''
#print(trainwin40)
#print(jocwin40)
# rule out nan
# change tname -> val
data = data_ori.dropna(axis=0, how='any')

tlst = data.tname.to_list()
t01 = list()
for i in tlst:
	if i in trainwin:
		t01.append(trainwin.get(i))
	else:
		t01.append(0)
data['tname'] = t01

# change jockey -> val
jlst = data.jname.to_list()
j01 = list()
for i in jlst:
	if i in jocwin:
		j01.append(jocwin.get(i))
	else:
		j01.append(0)
print(len(j01))
data['jname'] = j01



data.to_csv('data_dropnan.csv', index=False)


# split classes
# read in
df = pd.read_csv('data_dropnan.csv')

df_c5 = df[df['class'] == 'Class 5']
df_c4 = df[df['class'] == 'Class 4']
df_c3 = df[df['class'] == 'Class 3']
df_c2 = df[df['class'] == 'Class 2']
df_c1 = df[df['class'] == 'Class 1']


# output
df_c5.to_csv('data_class5.csv', index=False)
df_c4.to_csv('data_class4.csv', index=False)
df_c3.to_csv('data_class3.csv', index=False)
df_c2.to_csv('data_class2.csv', index=False)
df_c1.to_csv('data_class1.csv', index=False)