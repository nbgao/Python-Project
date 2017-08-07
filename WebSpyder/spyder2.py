# -*- coding: utf-8 -*-
## 伪装浏览器
import urllib.request

# 定义保存函数
def saveFile(data):
    path = "D:\\My_Document_New\\DataSource\\Douban\\douban_02.out"
    f = open(path,'wb')
    f.write(data)
    f.close()

# 网址
url = "https://www.douban.com/"
# 定义响应头
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '  
                        'Chrome/51.0.2704.63 Safari/537.36'}  

request = urllib.request.Request(url = url, headers = headers)
response = urllib.request.urlopen(request)
data = response.read()

# 保存爬取的数据至文件中
saveFile(data)

data = data.decode('utf-8')
print(data)

# 打印爬取网络的各类信息
print(type(response))
print(response.geturl())
print(response.info())
print(response.getcode())