#coding:utf-8
__author__ = 'dingrui'
import ana_1

#dd.download('https://movie.douban.com/subject/24773958/comments?start=')
(words_stat,words_df)=ana_1.jieba_ana()
#words_stat为关键词及权重数据帧 words_df为文章分词后的数据帧
ana_1.show(words_stat)