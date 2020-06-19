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

def in_parse_one_page(page):
    #查找出玩野中全部的 tr 标签并赋给 tr_list
    ul=driver.find_elements_by_class_name("rowlist")
    h3_list=ul[0].find_elements_by_tag_name("h3")
    return h3_list
def in_save_to_mysql(h3_list,filename):
    #创建一月的csv文件
    with open(filename,"a+",encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile,lineterminator='\n')
        for i in range(0, len(h3_list)):
            #找出 tr_list 中的全部 td 标签
            all=h3_list[i].find_elements_by_tag_name('a')
            print(all)
            str_url=[]
            str_url.append(all[0].text)
            print(str_url)
            writer.writerow(str_url)

def get_nation():
    filename="../data/good_url.csv"
    option = webdriver.ChromeOptions()
    driver = webdriver.Chrome()
    for i in range(100,101):
        url="http://www.alexa.cn/siterank/"+str(i)
        page = search(url)
        time.sleep(0.8)
        li_list = na_parse_one_page(page)
        na_save_to_mysql(li_list,filename)
        print(str(i)+"%")
    driver.close()   
    driver.quit()

def get_internation():
    global driver
    filename="../data/good_internation_url.csv"
    option = webdriver.ChromeOptions()
    driver = webdriver.Chrome()
    url="https://alexa.chinaz.com/Global/index.html"
    page = search(url)
    time.sleep(0.8)
    li_list = in_parse_one_page(page)
    in_save_to_mysql(li_list,filename)
    print(str(1))
    for i in range(2,21):
        url="https://alexa.chinaz.com/Global/index_"+str(i)+".html"
        page = search(url)
        time.sleep(0.8)
        li_list = in_parse_one_page(page)
        in_save_to_mysql(li_list,filename)
        print(str(i))
    driver.close()   
    driver.quit()
def main():
    #get_nation()
    get_internation()
    
if __name__=='__main__':
    main();  