#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

cardata = pd.read_csv('data/used_car_train_20200313.csv',sep=' ')
cartest = pd.read_csv('data/used_car_testB_20200421.csv',sep=' ')
# print(cardata.info())
# print(cardata.describe())
# print(cardata.head())

columns_dict={
    "SaleID":"交易ID",
    "name":"汽车交易名称",
    "regDate":"汽车注册日期",
    "model":"车型编码",
    "brand":"汽车品牌",
    "bodyType":"车身类型",
    "fuelType":"燃油类型",
    "gearbox":"变速箱",
    "power":"发动机功率",
    "kilometer":"汽车已行驶公里",
    "notRepairedDamage":"汽车有尚未修复的损坏",
    "regionCode":"地区编码",
    "seller":"销售方",
    "offerType":"报价类型",
    "creatDate":"汽车上线时间"
}


cardata.rename(columns = columns_dict,inplace=True)
cartest.rename(columns = columns_dict,inplace=True)

# 销售方
# 0    149999
# 1         1
# Name: 销售方, dtype: int64
# 报价类型
# 0    150000
# 删除以上 两项
cardata.drop(columns =["交易ID","汽车交易名称","销售方","报价类型"],inplace=True)
cartest.drop(columns =["交易ID","汽车交易名称","销售方","报价类型"],inplace=True)

cardata['地区编码'] = cardata['地区编码'].apply(lambda x:str(x)[:-3])
# print(cardata['地区编码'].value_counts())

tempcardata = cardata.copy()

target = tempcardata["price"]
tempcardata.drop(columns=["车型编码","汽车品牌","车身类型","燃油类型","变速箱","汽车有尚未修复的损坏","price"],inplace=True)

now = datetime.datetime.now().date()
def fuc(x,key):
    try:
        x=x.to_dict()
        year = str(x[key])[:4]
        month = str(x[key])[4:6]
        day = str(x[key])[6:8]
        return (now - datetime.date(int(year),int(month),int(day))).days
    except Exception as e:
        return  -1

tempcardata['汽车注册日期'] = tempcardata.apply(lambda x:fuc(x,'汽车注册日期'),axis=1)
tempcardata['汽车上线时间'] = tempcardata.apply(lambda x:fuc(x,'汽车上线时间'),axis=1)

cardata['地区编码'] = cardata['地区编码'].apply(lambda x:str(x)[:-3])
cartest['地区编码'] = cartest['地区编码'].apply(lambda x:str(x)[:-3])

dummies = pd.get_dummies(cardata['地区编码'],prefix='地区编码')
tempcardata = tempcardata.join(dummies)
dummies = pd.get_dummies(cardata['车型编码'],prefix='车型编码')
tempcardata = tempcardata.join(dummies)
dummies = pd.get_dummies(cardata['汽车品牌'],prefix='汽车品牌')
tempcardata = tempcardata.join(dummies)
dummies = pd.get_dummies(cardata['车身类型'],prefix='车身类型')
tempcardata = tempcardata.join(dummies)
dummies = pd.get_dummies(cardata['燃油类型'],prefix='燃油类型')
tempcardata = tempcardata.join(dummies)
dummies = pd.get_dummies(cardata['变速箱'],prefix='变速箱')
tempcardata = tempcardata.join(dummies)
dummies = pd.get_dummies(cardata['汽车有尚未修复的损坏'],prefix='汽车有尚未修复的损坏')
tempcardata = tempcardata.join(dummies)



print(tempcardata.info())

print(tempcardata.describe())
print(tempcardata.head(5))

# standardscaler = StandardScaler()
# minmaxscaler = MinMaxScaler()
# standardscaler.fit(tempcardata)
# newtempcardata = standardscaler.fit(tempcardata)
# minmaxscaler.fit(tempcardata)
# newtempcardata = minmaxscaler.fit(tempcardata)



def k_fold():
    from sklearn.model_selection import KFold
    import numpy as np

    kf = KFold(n_splits=5)
    train = np.array(tempcardata)
    target = np.array(target)

    for i,(train_index,test_index) in enumerate(kf.split(train)):
        print(i,train_index,test_index)
        print(train[train_index])
        print(target[train_index])





tempcardata['v_1'] = tempcardata['v_1'].apply(lambda x:abs(x),axis=1)

cols=["v_0","v_1","v_2","v_3","v_4","v_5","v_6","v_7","v_8","v_9","v_10","v_11","v_12","v_13","v_14"]
for i in range(len(cols)):
    plt.plot(tempcardata[cols[i]],target,'*')
    # tempcardata[cols[i]].hist(bins=500)
    plt.title(cols[i])
    plt.show()
    
print(cardata.columns)
# cardata[["v_0","v_1","v_2","v_3","v_4","v_5","v_6","v_7","v_8","v_9","v_10","v_11","v_12","v_13","v_14"]].plot.box()
cardata[cols].plot.box()
plt.grid(linestyle="--", alpha=0.3)
plt.show()
