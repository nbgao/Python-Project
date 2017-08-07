# -*- coding: utf-8 -*-
import urllib.request,socket,re,sys,os

# 定义文件保存路径
targetpath = "D:\\My_Document_New\\DataSource\\Douban\\douban_03_Images"

def saveFile(path):
    # 定义当前路径的有效性
    if not os.path.isdir(targetpath):
        os.mkdir(targetpath)
    
    # 设置每个图片的路径
    pos = path.rindex('/')
    t = os.path.join(targetpath,path[pos+1:])
    return t
    
# 网址
url = "http://www.douban.com"
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '  
                        'Chrome/51.0.2704.63 Safari/537.36'}  

request = urllib.request.Request(url = url,headers = headers)
response= urllib.request.urlopen(request)
data = response.read()

for link,t in set(re.findall(r'(https:[^s]*?(jpg|png|gif|bmp))',str(data))):
    print(link)
    try:
        urllib.request.urlretrieve(link,saveFile(link))
    except:
        print('Fail')
        

''' 
data = data.decode('utf-8')
saveFile(data)

print(data)

print(type(response))
print(response.geturl())
print(response.info())
print(response.getcode())
'''