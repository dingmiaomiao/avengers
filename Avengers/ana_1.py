import re
import jieba
import pandas as pd
from jieba_apply import jieba_seg
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # 词云包
import matplotlib
from jieba import analyse

# 引入TF-IDF关键词抽取接口

def jieba_ana():
    f = open('comments.txt', 'r', encoding='utf-8')
    pattern = re.compile(r'[\u4e00-\u9fa5]+')
    content = f.read()
    segment=jieba_seg(content)
    f1 = open(r'temp.txt', 'w')
    for wtemp in segment:
        f1.write(wtemp+' ')
    f1.close()
    f2 = open(r'temp.txt')
    temp_word=f2.read()
    word_list = re.findall(pattern, temp_word)
    words_df = pd.DataFrame({'word': word_list})
    #改进如何去除中文歧义
    keywords= jieba.analyse.extract_tags(temp_word, topK=101, withWeight=True)
    #print(keywords)#TFIDF算法权重自动自大到小排序
    stopwords = pd.read_csv("stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                            encoding='utf-8') #停用词
    str_stop=str(stopwords.stopword)
    words_stat = pd.DataFrame([t for t in keywords if t[0] not in str_stop], columns=['word', 'weights'])
    # print(words_stat)
    f.close()
    return (words_stat,words_df)

def show(words_stat):
    #词云表示
    matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
    wordcloud = WordCloud(font_path='hanyiqihei.ttf', background_color="white", max_font_size=80)  # 指定字体类型、字体大小和字体颜色
    word_frequence = {x[0]: x[1] for x in words_stat.head(1000).values}
    word_frequence_list = []
    for key in word_frequence:
        temp = (key, word_frequence[key])
        word_frequence_list.append(temp)

    wordcloud = wordcloud.fit_words(dict(word_frequence_list))
    plt.imshow(wordcloud)
    plt.show()
