import numpy as np
import os
import pandas as pd

# features: age(1), tname(1), jname(1), rating(1), track, bardraw(1), besttime(1), going

def t2s(t):
	t = str(t)
	if t != '0':
		m, s, ms = t.strip().split('.')
		ts = float(m) * 60 + float(s) + float(ms) * (1/100)
		return float(round(ts, 2))
	else:
		return 0

def repla_avg(l):
	l = np.asarray(l).astype('float')
	l[l == 0] = np.NaN
	avg = np.nanmean(l, axis=0)
	l[np.isnan(l)] = avg
	return l

#data = pd.read_csv('HR200709to201901.csv')
data = pd.read_csv('../Sample_test.csv')

#data['finishm'] = data['finishm'].replace(0, np.nan, inplace=True)
data = data.fillna(0)


# besttime  
lst = []
for column in data['besttime']:
	lst.append(t2s(column))
data['besttime'] = repla_avg(lst)



# quantify jname & tname
jockeydict = data.jname.value_counts().to_dict()
trainerdict = data.tname.value_counts().to_dict()

plapool = data[data['ind_pla'] == 1]
jockeywindict = plapool.jname.value_counts().to_dict()
trainerwindict = plapool.tname.value_counts().to_dict()

jocwin = dict()
for key, val in jockeydict.items():
	if key in jockeywindict:
		value = jockeywindict.get(key) / val
		if value < 0.6:
			jocwin.update({key: value})

trainwin = dict()
for key, val in trainerdict.items():
	if key in trainerwindict:
		value = trainerwindict.get(key) / val
		if value < 0.6:
			trainwin.update({key: value})

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
data['jname'] = j01
#print(data['jname'])

# track
trac = []
for i in data['track']:
	if i == 'TURF':
		trac.append(1)
	elif i == 'ALL WEATHER TRACK':
		trac.append(0)
	else:
		trac.append(None)
data['track'] = trac

#going
gol = []
for i in data['going']:
	if i == 'FAST':
		gol.append(1)
	elif i == 'GOOD':
		gol.append(2)
	elif i == 'GOOD TO FIRM':
		gol.append(3)
	elif i == 'GOOD TO YIELDING':
		gol.append(4)
	elif i == 'SLOW':
		gol.append(5)
	elif i == 'SOFT':
		gol.append(6)
	elif i == 'WET FAST':
		gol.append(7)
	elif i == 'WET SLOW':
		gol.append(8)
	elif i == 'YIELDING':
		gol.append(9)
	elif i == 'YIELDING TO SOFT':
		gol.append(10)
	else:
		gol.append(None)
data['going'] = gol
#print(data)

# finishm -> y

data.to_csv('sample.csv', encoding='utf-8', index=False)
print(len(data))
