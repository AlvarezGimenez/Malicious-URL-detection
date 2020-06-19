import csv
import pandas as pd
import numpy as np
import os
import time
from selenium import webdriver
def search(url):
    #利用get()方法获取网页信息并返回
    return driver.get(url)

def na_parse_one_page(page):
    #查找出玩野中全部的 tr 标签并赋给 tr_list
    ul=driver.find_elements_by_class_name("siterank-sitelist")
    li_list=ul[0].find_elements_by_tag_name("li")
    return li_list
def na_save_to_mysql(li_list,filename):
    #创建一月的csv文件
    with open(filename,"a+",encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile,lineterminator='\n')
        for i in range(0, len(li_list)):
            #找出 tr_list 中的全部 td 标签
            all=li_list[i].find_elements_by_class_name('domain')
            url=all[0].find_elements_by_tag_name('a')
            str_url=[]
            str_url.append(url[0].text)
            writer.writerow(str_url)
            
def get_nation():
    global driver
    filename="test1.csv"
    option = webdriver.ChromeOptions()
    driver = webdriver.Chrome()
    for i in range(0,50):
        url="https://tieba.baidu.com/f?kw=南昌大学&ie=utf-8&pn="+str(i)
        page = search(url)
        time.sleep(0.8)
        li_list = na_parse_one_page(page)
        na_save_to_mysql(li_list,filename)
        print(str(i)+"%")
    driver.close()   
    driver.quit()
def main():
    get_nation()
    
if __name__=='__main__':
    main();  