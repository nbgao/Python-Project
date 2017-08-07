# -*- coding: utf-8 -*-
#爬取豆瓣首页
import urllib.request

url = "http://www.douban.com"
# 请求
request = urllib.request.Request(url)
# 爬取结果
response= urllib.request.urlopen(request)

data = response.read()
#设置解码方式
data = data.decode('utf-8')

# 打印结果
print(data)

## 打印爬取网页的各类信息
# response类型
print(type(response))
# 获取url
print(response.geturl())
# 获取response基本信息
print(response.info())
# 获取响应码
print(response.getcode())