import re
import jieba
jieba.load_userdict("myself_dict.txt")

def jieba_seg(content):
    pattern = re.compile(r'[\u4e00-\u9fa5]+')
    filterdata = re.findall(pattern, content)#标点符号、空格等初步分句
    cleaned_comments = ''.join(filterdata)
    segment = jieba.lcut(cleaned_comments)#结巴中文分词
    return segment
