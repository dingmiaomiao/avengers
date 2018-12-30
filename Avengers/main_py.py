from keras.models import load_model
import pandas as pd  # 导入Pandas
import numpy as np  # 导入Numpy
import Model
from jieba_apply import jieba_seg
from pandas.core.frame import DataFrame


IS_TRAIN=False
PRE_PATH='comments.txt'

def predict(PRE_PATH):
    com= pd.read_table(PRE_PATH,header=None)
    words=com.values
    data_chars = []
    for temp_data in words:
        data_chars.append(list(jieba_seg(str(temp_data))))
    print(data_chars)
    data=Model.char2id(data_chars)
    print(data)
    data_last = np.array(data)
    model = load_model('model.h5')
    pre = model.predict_classes(data_last)
    (pos,neg)=(0,0)
    temp = {"comment":words.tolist() ,"label": pre.tolist()}
    result = DataFrame(temp)  # 将字典转换成为数据框
    print(result)
    f = open(r'result.txt', 'w')
    f.write(str(result))
    for i in pre:
        if i == 0:
            neg = neg + 1
        else:
            pos = pos + 1
    print('积极评论:',pos,'占总评论的',pos/(neg+pos))
    print('消极评论:',neg,'占总评论的',neg/(neg+pos))

if __name__ == "__main__":
    if IS_TRAIN:
        Model.train()
    else:
        predict(PRE_PATH)