import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from numpy import NaN

def t2s(t):
	t = str(t)
	if t != '0':
		m, s, ms = t.strip().split('.')
		ts = float(m) * 60 + float(s) + float(ms) * (1/100)
		return float(round(ts, 2))
	else:
		return 0


# m1
m1 = pd.read_csv('Week1Matchnew.csv', nrows=10).fillna(0)
l = list()
for column in m1['Besttime']:
	l.append(t2s(column))
m1['Besttime'] = l

col = ['Horseweight', 'Besttime']
m1[col] = m1[col].replace(0, NaN)
m1['Horseweight'] = m1['Horseweight'].fillna(m1['Horseweight'].mean())
m1['Besttime'] = m1['Besttime'].fillna(m1['Besttime'].mean())
#print(m1)

# m2
m2 = pd.read_csv('Week1Matchnew.csv', skiprows=12 ,nrows=12).fillna(0)
l1 = list()
for column in m2['Besttime']:
	l1.append(t2s(column))
m2['Besttime'] = l1

m2[col] = m2[col].replace(0, NaN)
m2['Horseweight'] = m2['Horseweight'].fillna(m2['Horseweight'].mean())
m2['Besttime'] = m2['Besttime'].fillna(m2['Besttime'].mean())
#print(m2)

# m3
m3 = pd.read_csv('Week1Matchnew.csv', skiprows=26 ,nrows=12).fillna(0)
l2 = list()
for column in m3['Besttime']:
	l2.append(t2s(column))
m3['Besttime'] = l2

m3[col] = m3[col].replace(0, NaN)
m3['Horseweight'] = m3['Horseweight'].fillna(m3['Horseweight'].mean())
m3['Besttime'] = m3['Besttime'].fillna(m3['Besttime'].mean())
#print(m3)

# m4
m4 = pd.read_csv('Week1Matchnew.csv', skiprows=40 ,nrows=12).fillna(0)
l3 = list()
for column in m4['Besttime']:
	l3.append(t2s(column))
m4['Besttime'] = l3

m4[col] = m4[col].replace(0, NaN)
m4['Horseweight'] = m4['Horseweight'].fillna(m4['Horseweight'].mean())
m4['Besttime'] = m4['Besttime'].fillna(m4['Besttime'].mean())
#print(m4)

# m5
m5 = pd.read_csv('Week1Matchnew.csv', skiprows=54 ,nrows=12).fillna(0)
l4 = list()
for column in m5['Besttime']:
	l4.append(t2s(column))
m5['Besttime'] = l4

m5[col] = m5[col].replace(0, NaN)
m5['Horseweight'] = m5['Horseweight'].fillna(m5['Horseweight'].mean())
m5['Besttime'] = m5['Besttime'].fillna(m5['Besttime'].mean())
#print(m5)

# m6
m6 = pd.read_csv('Week1Matchnew.csv', skiprows=68 ,nrows=12).fillna(0)
l5 = list()
for column in m6['Besttime']:
	l5.append(t2s(column))
m6['Besttime'] = l5

m6[col] = m6[col].replace(0, NaN)
m6['Horseweight'] = m6['Horseweight'].fillna(m6['Horseweight'].mean())
m6['Besttime'] = m6['Besttime'].fillna(m6['Besttime'].mean())
#print(m6)

# m7
m7 = pd.read_csv('Week1Matchnew.csv', skiprows=82 ,nrows=12).fillna(0)
l6 = list()
for column in m7['Besttime']:
	l6.append(t2s(column))
m7['Besttime'] = l6

m7[col] = m7[col].replace(0, NaN)
m7['Horseweight'] = m7['Horseweight'].fillna(m7['Horseweight'].mean())
m7['Besttime'] = m7['Besttime'].fillna(m7['Besttime'].mean())
#print(m7)

# m8
m8 = pd.read_csv('Week1Matchnew.csv', skiprows=96 ,nrows=12).fillna(0)
l7 = list()
for column in m8['Besttime']:
	l7.append(t2s(column))
m8['Besttime'] = l7

m8[col] = m8[col].replace(0, NaN)
m8['Horseweight'] = m8['Horseweight'].fillna(m8['Horseweight'].mean())
m8['Besttime'] = m8['Besttime'].fillna(m8['Besttime'].mean())
#print(m8)

# m9
m9 = pd.read_csv('Week1Matchnew.csv', skiprows=110 ,nrows=12).fillna(0)
l8 = list()
for column in m9['Besttime']:
	l8.append(t2s(column))
m9['Besttime'] = l8

m9[col] = m9[col].replace(0, NaN)
m9['Horseweight'] = m9['Horseweight'].fillna(m9['Horseweight'].mean())
m9['Besttime'] = m9['Besttime'].fillna(m9['Besttime'].mean())
#print(m9)
