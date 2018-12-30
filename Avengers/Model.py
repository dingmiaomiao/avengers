import os
import matplotlib.pyplot as plt
import gensim
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout
from keras.layers import  Embedding
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras import regularizers
from jieba_apply import jieba_seg
from sklearn import metrics
from keras.layers import Bidirectional
from keras.optimizers import Adam

MAX_LENGTH = 50 # 每条评论最大长度
EMBEDDING_DIM = 200 # 词向量空间维度
HIDDEN_UNITS = 128#隐藏层
DROPOUT_RATE = 0.5#丢失率
VALIDATION_SPLIT = 0.25 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例
VECTOR_DIR = 'baike.vectors.bin' # 词向量模型文件
WIKI_NAME='input.csv'#语料库的名字
WORD_INDEX_PATH='word_id_dict.txt'

def train_peration():
    dataset_path = './data'
    datafile = os.path.join(dataset_path,WIKI_NAME )
    raw_data = pd.read_csv(datafile)
    cln_data = raw_data.dropna().copy()#处理缺失数据
    # 建立新的一列，如果打分>=3.0，为正面评价1，否则为负面评价0
    cln_data['Tag'] = np.where(cln_data['Star'] >=3, 1, 0)
    cln_data=cln_data[['Comment','Tag']]#明显点赞数多的评论应该更被认同 占的权重大
    #print(cln_data.head())
    data_chars=[]
    for temp_data in cln_data['Comment']:
        data_chars.append(list(jieba_seg(temp_data)))
    labels=list(cln_data['Tag'])
    w = []  # 将所有词语整合在一起
    for line in data_chars:
        for word in line :
            w.append(word)
    dict1 = pd.DataFrame(pd.Series(w).value_counts(), columns=['sum'])  # 统计词的出现次数
    dict1['id'] = range(1, len(dict1) + 1)
    dict1.drop(['sum'], axis=1, inplace=True)
    dict1['word'] = dict1.index
    word_index = dict1.set_index('word').T.to_dict('list')
    with open(WORD_INDEX_PATH,'w') as f:
        f.write(str(word_index))
    data=char2id(data_chars)
    labs=[int(not x) for x in labels ]
    print(labels.count(1)/len(labels))
    #lab=np.array(labels)
    lab = np.array([labs,labels]).T
    data_last=np.array(data)
    return (data_last,lab,word_index)

def train():
    data_last,lab,word_index=train_peration()
    p1 = int(len(data_last) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    p2 = int(len(data_last) * (1 - TEST_SPLIT))
    x_train = data_last[:p1]
    y_train = lab[:p1]
    x_val = data_last[p1:p2]
    y_val = lab[p1:p2]
    x_test = data_last[p2:]
    y_test = lab[p2:]
    model = Sequential()
    model.add(emb_myself(word_index))#使用自定义的嵌入层
    #bilstm层 并设置dropout
    model.add(Bidirectional(LSTM(HIDDEN_UNITS,recurrent_dropout=0.2,return_sequences = True)))
    model.add(Dropout(DROPOUT_RATE))  # 防止过拟合
    model.add(Bidirectional(LSTM(HIDDEN_UNITS,recurrent_dropout=0.2)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(lab.shape[1], activation='sigmoid', kernel_regularizer=regularizers.l2(1)))
    model.summary()#模型打印
    # 编译模型  优化器optimizer，损失函数loss，评估指标metrics  binary_crossentropy只能用于二分类输出
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=128)
    model.save('model.h5')
    print(model.evaluate(x_test, y_test))  # loss accuary
    y_pred= model.predict_classes(x_test)
    print(metrics.recall_score(y_test.T[1], y_pred, average='micro'))#召回率
    print(metrics.f1_score(y_test.T[1], y_pred, average='weighted'))#F值
    training_vis(history)

def emb_myself(word_index):#使用word2vec替换原嵌入层
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(VECTOR_DIR, binary=True)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if str(word) in w2v_model:
            embedding_matrix[i] = np.asarray(w2v_model[str(word)], dtype='float32')
    embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                                input_length=MAX_LENGTH, trainable=False)
    return embedding_layer

def char2id(data_chars):
    with open(WORD_INDEX_PATH, 'r+') as fr:
        word_index = eval(fr.read())  # 读取的str转换为字典
    data = data_chars[:]
    i = 0
    for line in data_chars:
        temp = []
        if MAX_LENGTH<len(line):
            j=0
            for word in line:
                if j < MAX_LENGTH:
                    if word not in word_index:
                        temp.append(0)
                    else:
                        temp.append(word_index[word][0])
                    j=j+1
                else:
                    break
        else:
            for word in line:
                if word not in word_index:
                    temp.append(0)
                else:
                    temp.append(word_index[word][0])
            for j in range(0,MAX_LENGTH-len(line)):
                temp.append(0)
        data[i] = temp
        i = i + 1
    return data

def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc,label='train_acc')
    ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.legend()
    plt.tight_layout()
    plt.show()
