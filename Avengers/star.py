from urllib import request
from bs4 import BeautifulSoup as bs

headers = {'User-Agent':'User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'}


for i in range(0, 11):
    res = request.urlopen('https://movie.douban.com/subject/24773958/comments?start=' + 
    str(20 * i) + '&limit=20&sort=new_score&status=P&percent_type=')
    html_data = res.read().decode('utf-8')
    Soup = bs(html_data, 'html.parser')

    for j in Soup.find_all('span',attrs={'class':'rating'}):
        star_text=j.get('class')[0][7]+'\n'
        f = open('star.txt', 'a', encoding='utf-8')
        f.write(star_text)
        f.close()

    for k in Soup.find_all('span',attrs={'class':'comment-time '}):
        time=k.get('title')+'\n'
        f2 = open('time.txt', 'a', encoding='utf-8')
        f2.write(time)
        f2.close()


    
