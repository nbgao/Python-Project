# python爬虫爬取豆瓣电影TOP100的电影名称

import requests
from requests.exceptions import RequestException
import re
import json


def get_one_page(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        return None
    except RequestException:
        return None


def parse_one_page(html):
    pattern = re.compile('<table width=".*?<div class="pl2">.*?>(.*?)</a>.*?class="pl">(.*?)</p>'
                         + '.*?<span class="rating_nums">(.*?)</span>.*?class="pl">(.*?)</span>', re.S)
    items = re.findall(pattern, html)
    for item in items:
        yield{
            'title': item[0].split("/")[0],
            'time': item[1].split("/")[0],
            'actor': item[1].split("/")[1:],
            'average': item[2],
            'content': item[3],
        }


def write_to_file(content):
    with open('movie2016.txt', 'a', encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False) + '\n')
        f.close()


def main(start):
    url = 'https://movie.douban.com/tag/2016?start=' + str(start) + '&type=T'
    html = get_one_page(url)
    for item in parse_one_page(html):
        print(item)
        write_to_file(item)

if __name__ == '__main__':
    for i in range(0, 10):
        main(i*20)

