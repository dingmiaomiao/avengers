import csv

need=list(range(1,530))
num=[]
try:
    with open('./data/DMSC.csv', 'r') as db01:
    #返回一个生成器对象，reader是可迭代的
        reader = csv.reader(db01)
        i=1
        for row in reader:
            if i in need:
                num.append(row)
                i=i+1
#捕捉异常本身，打印异常信息
except csv.Error as e:
  print("Error at line %s :%s", reader.line_num, e)
with open('train_yuliao.csv','w') as f:
    f.write(str(num))
