import gensim
import pandas as pd
import numpy as np
import os
from jieba_apply import jieba_seg
from Model import train_peration
from sklearn.svm import SVC
import Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

#读入数据（评论集与标签集）
data_last,lab,word_index =train_peration()
lab_last=lab.T[1]
#切片划分为训练集、验证集、测试集
x_train=data_last[:30000]
y_train=lab_last[:30000]
x_val=data_last[30000:40000]
y_val=lab_last[30000:40000]
x_test=data_last[40000:]
y_test=lab_last[40000:]

print('逻辑回归:')
lr_model = LogisticRegression(penalty='l1',class_weight={0:0.3,1:0.7},solver="liblinear")
lr_model.fit(x_train, y_train)#训练集
print("val mean accuracy: {0}".format(lr_model.score(x_val, y_val)))#验证集
y_pred = lr_model.predict(x_test)#测试集
print(classification_report(y_test, y_pred))

print('随机森林:')
#n_estimators 森林里（决策）树的数目
#criterion 计算属性的gini(基尼不纯度)还是entropy(信息增益)
#random_state 随机数生成器使用的种子 整数
#max_features 允许单个决策树使用特征的最大数量 sqrt每颗子树可以利用总特征数的平方根个 防止过拟合
rf_model = RandomForestClassifier(n_estimators=300,criterion='gini' ,
                                  max_features='sqrt',random_state=100)
rf_model.fit(x_train, y_train)#训练集
print("val mean accuracy: {0}".format(rf_model.score(x_val, y_val)))#验证集
y_pred = rf_model.predict(x_test)#测试集
print(classification_report(y_test, y_pred))




