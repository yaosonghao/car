#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime,time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

cardata = pd.read_csv('data/used_car_train_20200313.csv',sep=' ')
cartest = pd.read_csv('data/used_car_testB_20200421.csv',sep=' ')

# cardata.dropna(inplace=True)
# cartest.dropna(inplace=True)

cartestSaleID = cartest['SaleID']
cartestSaleID.astype(str)
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


# # 去除异常值
# from utils import outliers_proc
# cols=["v_0","v_1","v_2","v_3","v_4","v_5","v_6","v_7","v_8","v_9","v_10","v_11","v_12","v_13","v_14"]
# for i in range(len(cols)):
#     cardata = outliers_proc(cardata,cols[i],3)
 



# 销售方
# 0    149999
# 1         1
# 报价类型
# 0    150000
# 删除以上 两项 
# "交易ID","汽车交易名称" 无关项删除
cardata.drop(columns =["交易ID","汽车交易名称","销售方","报价类型"],inplace=True)
cartest.drop(columns =["交易ID","汽车交易名称","销售方","报价类型"],inplace=True)

tempcardata = cardata.copy()
tempcartest = cartest.copy()

target = tempcardata["price"]
tempcardata.drop(columns=["车型编码","汽车品牌","车身类型","燃油类型","变速箱","汽车有尚未修复的损坏","price"],inplace=True)
tempcartest.drop(columns=["车型编码","汽车品牌","车身类型","燃油类型","变速箱","汽车有尚未修复的损坏"],inplace=True)
now = datetime.datetime.now().date()
def fuc(x,key):
    try:
        x=x.to_dict()
        year = str(x[key])[:4]
        month = str(x[key])[4:6]
        day = str(x[key])[6:8]
        return (now - datetime.date(int(year),int(month),int(day))).days
    except Exception as e:
        # print(key,x[key])
        return  -1

tempcardata['汽车注册日期'] = tempcardata.apply(lambda x:fuc(x,'汽车注册日期'),axis=1)
tempcardata['汽车上线时间'] = tempcardata.apply(lambda x:fuc(x,'汽车上线时间'),axis=1)
tempcartest['汽车注册日期'] = tempcartest.apply(lambda x:fuc(x,'汽车注册日期'),axis=1)
tempcartest['汽车上线时间'] = tempcartest.apply(lambda x:fuc(x,'汽车上线时间'),axis=1)


cardata['地区编码'] = cardata['地区编码'].apply(lambda x:str(x)[:-3])
cartest['地区编码'] = cartest['地区编码'].apply(lambda x:str(x)[:-3])


dummies = pd.get_dummies(cardata['地区编码'],prefix='地区编码')
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

dummies = pd.get_dummies(cartest['地区编码'],prefix='地区编码')
tempcartest = tempcartest.join(dummies)
dummies = pd.get_dummies(cartest['汽车品牌'],prefix='汽车品牌')
tempcartest = tempcartest.join(dummies)
dummies = pd.get_dummies(cartest['车身类型'],prefix='车身类型')
tempcartest = tempcartest.join(dummies)
dummies = pd.get_dummies(cartest['燃油类型'],prefix='燃油类型')
tempcartest = tempcartest.join(dummies)
dummies = pd.get_dummies(cartest['变速箱'],prefix='变速箱')
tempcartest = tempcartest.join(dummies)
dummies = pd.get_dummies(cartest['汽车有尚未修复的损坏'],prefix='汽车有尚未修复的损坏')
tempcartest = tempcartest.join(dummies)



# 对 v_1 求绝对值
tempcardata['v_1'] = tempcardata['v_1'].apply(lambda x:abs(x))
tempcartest['v_1'] = tempcartest['v_1'].apply(lambda x:abs(x))
# 对 v_2 大于10的减15
tempcardata['v_2'] = tempcardata['v_2'].apply(lambda x:abs(x))
tempcartest['v_2'] = tempcartest['v_2'].apply(lambda x:abs(x))

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

standardscaler = StandardScaler()
minmaxscaler = MinMaxScaler()

standardscaler.fit(tempcardata)
newtempcardata = standardscaler.transform(tempcardata)
newtempcartest = standardscaler.transform(tempcartest)

minmaxscaler.fit(newtempcardata)
newtempcardata = minmaxscaler.transform(newtempcardata)
newtempcartest = minmaxscaler.transform(newtempcartest)

print(newtempcardata.shape)
print(newtempcartest.shape)



def LR():
    from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
    import numpy as np
    model = linear_model.LinearRegression()
    model.fit(newtempcardata, target)
    print(model.intercept_)  #截距
    print(model.coef_)  #线性模型的系数
    a = model.predict(newtempcardata)
    b = model.predict(newtempcartest)
    df = pd.concat([cartestSaleID,pd.Series(b)],ignore_index=True,axis=1)
    df.rename(columns = {0:"SaleID",1:"price"},inplace=True)
    df['price'] = df['price'].apply(lambda x:x if x>0 else 0) 
    df.to_csv('res'+str(int(time.time()))+'.csv',index=False)

def RF():
    ####随机森林回归####
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=800)   #esitimators决策树数量
    model.fit(newtempcardata, target)
    a = model.predict(newtempcardata)
    b = model.predict(newtempcartest)
    df = pd.concat([cartestSaleID,pd.Series(b)],ignore_index=True,axis=1)
    df.rename(columns = {0:"SaleID",1:"price"},inplace=True)
    df['price'] = df['price'].apply(lambda x:x if x>0 else 0) 
    df.to_csv('res'+str(int(time.time()))+'.csv',index=False)


# RF()

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor

def model_m(newtempcardata,target):
    X,X_test,Y,Y_test = train_test_split(newtempcardata,target,test_size=0.1)
    print(X.shape)
    print(Y.shape)
    print(X_test.shape)
    print(Y_test.shape)
    models = [XGBRegressor(n_estimators = 500, objective='reg:squarederror'),RandomForestRegressor(n_estimators=500)]
    resulttrain = []
    resulttest = []
    result  = []
    for model in models:
        model.fit(X, Y)
    for model in models:
        resulttrain.append(model.predict(X))
        resulttest.append(model.predict(X_test))
        result.append(model.predict(newtempcartest))

    dftrain = pd.concat([pd.Series(resulttrain[0]),pd.Series(resulttrain[1])],ignore_index=True,axis=1)
    dftest = pd.concat([pd.Series(resulttest[0]),pd.Series(resulttest[1])],ignore_index=True,axis=1)
    df = pd.concat([pd.Series(result[0]),pd.Series(result[1])],ignore_index=True,axis=1)
    
    print(np.mean(abs(resulttrain[0] - target)))
    print(np.mean(abs(resulttrain[1] - target)))

    dftrain.rename(columns = {0:"price1",1:"price2"},inplace=True)
    model = linear_model.LinearRegression()
    model_stack.fit(dftrain,target)

    print(np.mean(abs(model_stack.predict(dftrain) - Y)))
    print(np.mean(abs(model_stack.predict(dftest) - Y_test)))



    model_predict = model_stack.predict(df)
    dfres = pd.concat([cartestSaleID,pd.Series(model_predict)],ignore_index=True,axis=1)

    dfres.rename(columns = {0:"SaleID",1:"price"},inplace=True)
    dfres['price'] = dfres['price'].apply(lambda x:x if x>0 else 0) 
    dfres.to_csv('model_mres'+str(int(time.time()))+'.csv',index=False)


model_m(newtempcardata,target)