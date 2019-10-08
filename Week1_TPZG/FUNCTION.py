
import os
import numpy as np
import pandas as pd
import statistics
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


from readin import m1,m2,m3,m4,m5,m6,m7,m8,m9

data1 = pd.read_csv('test_data2.csv')

# print(trainwin40)


def change(data,x):
    # trainer -> 1,0
    trainerdict = data1.tname.value_counts().to_dict()
    trainerwindict = data1.tname.value_counts().to_dict()

    # trainer win_odds
    trainwin = dict()
    for key, val in trainerdict.items():
        if key in trainerwindict:
            value = trainerwindict.get(key) / val
            trainwin.update({key: value})

    # 40 percentile of trainer
    tra = [v for k, v in trainwin.items()]
    tra40 = np.percentile(tra, 40, axis=0)
    # print(tra40)

    # trainer the first 40% winning odds mark as 1, else: 0
    trainwin40 = dict()
    for k, v in trainwin.items():
        if v >= tra40:
            trainwin40.update({k: 1})
        else:
            trainwin40.update({k: 0})

    jockeydict = data1.jname.value_counts().to_dict()
    jockeywindict = data1.jname.value_counts().to_dict()

    jocwin = dict()
    for key, val in jockeydict.items():
        if key in jockeywindict:
            value = jockeywindict.get(key) / val
            jocwin.update({key: value})

    # 40 percentile of jockey
    joc = [v for k, v in jocwin.items()]  # convert to a list
    joc40 = np.percentile(joc, 40, axis=0)

    # jockey win_odds

    # jockey the first 40% winning odds mark as 1, else: 0
    jocwin40 = dict()
    for k, v in jocwin.items():
        if v >= joc40:
            jocwin40.update({k: 1})
        else:
            jocwin40.update({k: 0})



    # hid -> 1,0
    data["hid"] = data["hid"].apply(lambda x: 1*0.19 if 1 <= x <= 8 else 0)

    # trainer -> 1,0
    data["Trainer"] = data["Trainer"].apply(lambda x: 1*0.1 if x in trainwin40 else 0)

    # jokey -> 1,0
    data["Jockey"] = data["Jockey"].apply(lambda x: 1*0.15 if x in jocwin40 else 0)


    # raiting -> 1,0 : 40 percentile
    rating_list = data["Rate"].tolist()
    rating40 = np.percentile(rating_list, 40, axis=0)

    data["Rate"] = data["Rate"].apply(lambda x: 1 *0.08if x > rating40 else 0)

    # ratechange -> positive:1, negative:0
    data["Rtg"] = data["Rtg"].apply(lambda x: 0 if x == np.nan else x)
    data["Rtg"] = data["Rtg"].apply(lambda x: 1*0.08 if x >= 0 else 0)

    # weight -> 1,0
    data["Horseweight"] = data["Horseweight"].apply(lambda x: 1111.7815 if x == np.nan else x)
    data["Horseweight"] = data["Horseweight"].apply(lambda x: 1* 0.08 if 1041.497515 < x < 1182.066201 else 0)

    # Besttime -> 1, 0 /

    k=data["Besttime"].tolist()
    k_mean= statistics.mean(k)


    data["Besttime"]=data["Besttime"].apply(lambda x: (x-k_mean)/k_mean)

    k=data["Besttime"].tolist()
    besttime30 = np.percentile(k, 30, axis=0)

    data["Besttime"] = data["Besttime"].apply(lambda x: 1 *0.32 if x <= besttime30 else 0)

    data["RowSum"] = data.sum(axis=1)
    sum_d=data["RowSum"].sum(axis=0)

    data["RowSum"] = data["RowSum"].apply(lambda x: x/sum_d)
    R = data["RowSum"].tolist()
    R.sort(reverse=True)


    first = R[0]
    first3 = R[0:3]

    k = list()
    for i in data["RowSum"]:
        if i == first:
            k.append(1)
        else :
            k.append(0)

    l= list()
    for i in data["RowSum"]:
        if i in first3:
            l.append(1)

        else:
            l.append(0)

    data["first"]=k
    data["place"]=l

    data["winstake"] = data["first"].apply(lambda x: x * 10)
    data["plastake"] = data["place"].apply(lambda x: x * 10)

    data.to_csv("test_recent_check.csv")

    Sub=data[["first","place","winstake","plastake"]]

    print(Sub)

    name="match"+str(x)
    data.to_csv(name+".csv")



change(m1,1)
change(m3,3)
change(m4,4)
change(m5,5)
change(m6,6)
change(m7,7)



