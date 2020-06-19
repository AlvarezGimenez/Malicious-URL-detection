from xml.dom.minidom import parse
import urllib.request
import requests
from bs4 import BeautifulSoup
import json
def readhtml(url):#url转换为html格式
    head={}
    data={}
    head['User-Agent']="Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36"
    req=urllib.request.Request(url,data,head)
    response=urllib.request.urlopen(req)
    html=response.read()
    html=html.decode('utf-8')
    file=open(r"../data/xml.txt","w",encoding='utf-8')
    file.writelines(html);
    file.close();
   
def readjson():
    f =open('../data/html.json',encoding='utf-8') #打开‘product.json’的json文件
    res=f.read()  #读文件
    result=json.loads(res)
    url_list=[]
    for i in result:
        url_list.append(i['url'])
    print(len(url_list))
    print(url_list)
    
def main():
    url="http://data.phishtank.com/data/822c4203632b270307eb0f82e381293d16357f4cdc5b1efb3f33c8533b72c682/online-valid.json"
    #html=readhtml(url)
    #print(html)
    readjson()
    
if __name__=='__main__':
    main();    