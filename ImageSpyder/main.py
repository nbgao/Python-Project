import re
import requests
import os
import time
import math

global cnt

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if isExists:
        # print(path + '目录已存在')
        return False
    else:
        os.makedirs(path)
        print('目录 ' + path + ' 创建成功')
        return True

def downloadPicture(html, keyword, num):
    global cnt

    path = './Images/' + keyword
    # 若目录不存在则创建新目录
    mkdir(path)
    pic_url = re.findall('"objURL":"(.*?)"', html, re.S)
    # cnt = 1

    print('[INFO]:找到关键词：' + keyword + '的图片，现在开始下载图片...')
    for url in pic_url:
        print('[INFO]:正在下载第' + str(cnt+1) + '张图片，图片链接：' + str(url))
        try:
            pic = requests.get(url, timeout=10)
        except requests.exceptions.ConnectionError:
            print('[ERROR]:当前图片无法下载.')
            continue

        dir = './Images/' + keyword + '/' + "{:0>6d}".format(cnt+1) + '.jpg'
        fp = open(dir, 'wb')
        fp.write(pic.content)
        fp.close()
        cnt += 1
        if cnt >= num:
            print('[Completed]:已成功爬取' + str(num) + '张' + keyword + '图片.\n')
            return num

    # print('[Completed]:已成功爬取' + str(cnt-1) + '张' + keyword + '图片.')
    return cnt-1

'''
1 坦克
2 装甲车
3 军舰
4 航空母舰
5 潜艇
6 战斗机
7 直升机
8 轰炸机
9 士兵
'''
if __name__ == '__main__':
    start_time = time.time()
    dict = ['坦克','装甲车','军舰','航空母舰','潜艇','战斗机','直升机','轰炸机','士兵']
    dict = ['战斗机', '直升机', '轰炸机', '士兵']
    # while True:
    #     keyword = input('[INPUT]:请输入需要爬取图片的关键字：')
    #     num = int(input('[INPUT]:请输入需要爬取图片的数量：'))
    for keyword in dict:
        num = 120

        cnt = 0
        page = 1

        # url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + keyword + '&ct=201326592&v=flip'
        for page in range(0,int(math.ceil(num/60)+1)):
            if cnt >= num:
                break

            url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + keyword + 'pn=' + str(60*page) + '&ct=&ic=0&lm=-1&width=0&height=0'

            t1 = time.time()
            result = requests.get(url)
            t2 = time.time()
            print('[INFO]:Page' + str(page+1) + '解析耗时：{:.3f}s'.format(t2-t1))

            downloadPicture(result.text, keyword, num)
            t3 = time.time()
            print('[INFO]:Page' + str(page+1) + '图片下载耗时：{:.3f}s'.format(t3-t2))

    end_time = time.time()
    print('[INFO]:总运行时间：{:.3f}s\n'.format(end_time - start_time))