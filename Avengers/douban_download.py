from urllib import request
from bs4 import BeautifulSoup as bs

def download(obj_str):
    for i in range(0, 11):#多少页评论
        res = request.urlopen( obj_str+ str(
            20 * i) + '&limit=20&sort=new_score&status=P&percent_type=')
        html_data = res.read().decode('utf-8')
        Soup = bs(html_data, 'html.parser')
        comments = Soup.find_all('div', id='comments')
        comments_content = comments[0].find_all('p')
        for j in range(0, 20):#每页20个评论
            text = str(comments_content[j])
            f = open('comments.txt', 'a', encoding='utf-8')
            f.write(text)
            f.close()