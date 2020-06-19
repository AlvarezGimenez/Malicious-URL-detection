import urllib.request
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import csv
import pandas as pd
import numpy as np
import os
import time
import datetime
import re
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab 
import matplotlib.dates as mdate
import string
import whois as wl
from urllib.parse import urlparse
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  , TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
def search(url):
    #获得网页的html
    return driver.get(url)
def parse_one_page(page):
    #根据网页的html提取标签中的数据
    a_list=driver.find_elements_by_tag_name("a")
    return a_list
def save_to_mysql(a_list,filename):
    #储存数据至csv文件
    with open(filename,"a+",encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile,lineterminator='\n')
        for i in range(0, len(a_list)):
            all=a_list[i].get_attribute("href")
            str_url=str(all)
            if(str_url.strip()==''or str_url=="None"):
                continue
            rule=re.compile(r'javascript.')
            if(re.search(rule,str_url)):
                continue
            else:
                if(str_url=="chrome-error://chromewebdata/#buttons"):
                    continue
                print(str_url)
                url_list=[]
                url_list.append(str_url)
                writer.writerow(url_list)
def get_good_urls(path):
    #获取非恶意网站的urls
    datas=[]
    global driver
    option = webdriver.ChromeOptions()
    driver = webdriver.Chrome()
    with open(path,'r',encoding='utf-8') as lines:
        for line in lines:
            datas.append(line.strip())
    filename="../data/good_urls.csv"
    count=0
    for i in datas:
        count=count+1
        if(count<362):
            continue
        try:
            try:
                page = search("https://"+i)
            except Exception:
                page = search("http://"+i)
            finally:
                a_list = parse_one_page(page)
                save_to_mysql(a_list,filename)
        except Exception:
            print(i+"没有数据")
def deal_good_urls():
    #处理良好的urls
    filename="../data/good_urls.csv"
    datas=pd.read_csv(filename,skip_blank_lines=False)
    urls=list(datas['https://www.baidu.com/'])
    
    #过滤非正常网页格式（http，https）
    rule=re.compile(r'^https?.')
    datas=[]
    for url in urls:
        if(re.search(rule,url)):
            datas.append(url)
    
    #消除重复的url
    all={}
    for data in datas:
        if(data not in all.keys()):
            all[data]=1
    datas=[]
    datas=list(all.keys())
    data_len=[]
    
    for data in datas:
        data_len.append(len(data))
    
    '''
    plt.boxplot(data_len)
    plt.show()
    '''
    '''
    rule=re.compile(r'(com/?$){1,1}')
    l=[]
    for data in datas:
        if(re.search(rule,data)):
            l.append(data)
    '''
    print(len(datas))
    return datas 
    
def get_bad_urls(path):
    #获取恶意网页的urls
    datas=[]
    global driver
    #option = webdriver.ChromeOptions()
    #driver = webdriver.Chrome()
    all=pd.read_csv(path,skip_blank_lines=False)
    datas=list(all['url'])
    '''
    rule=re.compile(r'(\.com/{0,1}){1,1}$')
    for data in datas:
        if(re.search(rule,data)):
            print(data)
    '''
    data_len=[]
    for data in datas:
        data_len.append(len(data))
    '''
    plt.boxplot(data_len)
    plt.show()
    '''
    print(len(datas))
    return datas
     #filename="../data/bad.csv"
def bad_merge():
    #每次合并更新的恶意urls数据，在add_name处输入文件名，即可合并
    main_name='../data/bad.csv'
    add_name='../data/43.csv'
    add_data=pd.read_csv(add_name,skip_blank_lines=False)
    main_data=pd.read_csv(main_name,skip_blank_lines=False)
    main_id_datas=list((main_data['phish_id']))
    #for index,row in list.iterrows():
    with open(main_name,"a+",encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile,lineterminator='\n')
        for index,i in add_data.iterrows():
            if((int)(i['phish_id']) in main_id_datas):
                break
            else:
                l=[]
                for x in i:
                    l.append(x)
                writer.writerow(l)
    main_data=pd.read_csv(main_name,skip_blank_lines=False)
    print(main_data)
    
def same(data1,data2):
    #使得data1，data2数据数目相同
    num = random.sample(range(1, len(data1)), len(data2))
    data=[]
    for i in num:
        data.append(data1[i])
    return data
def get_features(data1,data2,good,bad):
    #提取特征（训练数据提取）
    #注：如果模型测试时，一个
    if os.path.exists("../data/all_features.csv"):  # 如果文件存在
    # 删除文件，可使用以下两种方法。
        os.remove("../data/all_features.csv")  
    #1.获取urls长度特征
    print("长度："+str(len(data1))+" "+str(len(data2)))
    good_len=[]
    bad_len=[]
    total=data1+data2
    
    for i in data1:
        good_len.append(len(i))
    for j in data2:
        bad_len.append(len(j))
    datas=pd.DataFrame()
    datas['url']=total
    datas['length']=good_len+bad_len
    
    #2。恶意符号数目
    evil_sign=['~','!','@','#','$','^','*','-']
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if j in evil_sign:
                count=count+1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['evil_sign']=gl+bl
    
    #3。 。的个数
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j=='.'):
                count=count+1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['point']=gl+bl
    
    #4。 /的个数
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j=='/'):
                count=count+1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['class']=gl+bl
    
    #5.数字占有比例
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j.isdigit()):
                count=count+1
        if(number<=len(data1)):
            gl.append((float)(count/len(i)))
        else:
            bl.append((float)(count/len(i)))
    datas['number']=gl+bl
    
    
    #6.字母占有比例
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j.isalpha()):
                count=count+1
        if(number<=len(data1)):
            gl.append((float)(count/len(i)))
        else:
            bl.append((float)(count/len(i)))
    datas['alpha']=gl+bl
    
    
    #7.特殊字符收尾
    tail_sign=['=','_','.','\\','+','~']
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        if(i[-1] in tail_sign):
            count=1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)

    datas['tail_sign']=gl+bl
    
    #8.大写字母数量
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j.isupper()):
                count=count+1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['upper']=gl+bl
    
    #9.?=&三种符号的关系
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=1
        num1=0
        num2=0
        num3=0
        for j in i:
            if(j=='?'):
                num1=num1+1
            if(j=='='):
                num2=num2+1
            if(j=='&'):
                num3=num3+1  
        if(num1==0):
            if(num2==0 and num3==0):
                count=0
        else:
            if(num2>=1 and num3>=0 and num3<=num2-1):
                count=0
        if(num1>1):
            count=1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['?=&']=gl+bl
    
    #10.是否存在IP地址
    rule="\W((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}\W"
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        if(re.search(rule,i)):
            count=1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['IP']=gl+bl
    
    #11.统计元音和辅音的比例
    voice=['a','e','o','i','u']
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        yes=0
        for j in i:
            if(j.isalpha()):
                count=count+1
                if(j in voice):
                    yes=yes+1
        if(number<=len(data1)):
            gl.append(float(yes/count))
        else:
            bl.append(float(yes/count))
    datas['voice']=gl+bl
    
    #12.域名级数
    gl=[]
    bl=[]
    number=0
    for i in total:
        goal=urlparse(i).netloc#提取域名
        number=number+1
        count=0
        for j in goal:
            if(j=='.'):
                count=count+1
        if(number<=len(data1)):
            gl.append(count+1)
        else:
            bl.append(count+1)
    datas['domain_num']=gl+bl
    
    #13.最长域名段长度
    gl=[]
    bl=[]
    number=0
    for i in total:
        goal=urlparse(i).netloc#提取域名
        number=number+1
        max=0
        str_list=goal.split('.')
        for j in str_list:
            if(max<=len(j)):
                max=len(j)
        if(number<=len(data1)):
            gl.append(max)
        else:
            bl.append(max)
    datas['domain_max_len']=gl+bl
    
    #14.连续数字最大长度
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        max=0
        flag=0
        l=0
        for j in i:
            if(j.isdigit()):
                flag=1
            else:
                flag=0
                l=0
            if(flag==1):
                l=l+1
            if(l>=max):
                max=l
        if(number<=len(data1)):
            gl.append(max)
        else:
            bl.append(max)
    datas['max_num_len']=gl+bl
    '''
    g=[]
    for i in range(1,len(data1)+1):
        g.append(i)
    plt.scatter(gl,g,color="green")
    plt.scatter(bl,g,color="red")
    plt.show()
    '''
    
    
    #15.tf-idf断词,统计词语数量
    gl_g=[]
    bl_g=[]
    gl_b=[]
    bl_b=[]
    good,bad=tf_idf(data1,data2)
    good_words=(list)(good.keys())
    bad_words=(list)(bad.keys())
    number=0
    for i in total:
        number=number+1
        #将文本中的词语转换为词频矩阵  
        vectorizer = CountVectorizer()  
        #计算个词语出现的次数  
        s=[]
        s.append(i)
        X = vectorizer.fit_transform(s)  
        #获取词袋中所有文本关键词  
        words = vectorizer.get_feature_names()  
        g_number=0
        b_number=0
        for word in words:
            if word in good_words:
                g_number=g_number+good[word]
            if word in bad_words:
                b_number=b_number+bad[word]
        if(number<=len(data1)):
            gl_b.append(b_number)
            gl_g.append(g_number)
        else:
            bl_b.append(b_number)
            bl_g.append(g_number)
    datas['good_word']=gl_g+bl_g
    datas['bad_word']=gl_b+bl_b
    
    #16.whois注册时间差
    #注：训练模型时使用
    gl=[]
    bl=[]
    number=0
    times=pd.read_csv("../data/time.csv")
    all=times.set_index('url').T.to_dict('list')
    for i in total:
        number=number+1
        name=urlparse(i).netloc
        print(number)
        print(i)
        time=all[name][0]
        x=0
        if(time>0 and time<3):
            x=1
        if(number<=len(data1)):
            gl.append(x)
        else:
            bl.append(x)
    datas['time']=gl+bl
    
    
    evil=[]
    for i in data1:
        evil.append(0)
    for i in data2:
        evil.append(1)
    datas['evil']=evil
    '''
    datas.to_csv("../data/all_features.csv",index=0)
    '''
    return datas

def tf_idf(data1,data2):
    #获取tf-idf的参数
    vectorizer_good=TfidfVectorizer(min_df=0.27)#过滤小于0.27的词汇
    g_tfidf=vectorizer_good.fit_transform(data1)
    gl_word=vectorizer_good.get_feature_names()
    gl_weight=g_tfidf.nonzero()
    row=(list)(gl_weight[0])
    column=(list)(gl_weight[1])
    good_dict={}
    for i in range(len(row)):
        if gl_word[column[i]] not in good_dict.keys():
            good_dict[gl_word[column[i]]]=g_tfidf[row[i],column[i]]
        else:
            good_dict[gl_word[column[i]]] += float(g_tfidf[row[i],column[i]])
        
    sorted_good_tfidf = sorted(good_dict.items(), 
                      key=lambda d:d[1],  reverse = True )
    middle_good=(dict)(sorted_good_tfidf)    
    vectorizer_bad=TfidfVectorizer(min_df=0.27)#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频 
    b_tfidf=vectorizer_bad.fit_transform(data2)
    bl_word=vectorizer_bad.get_feature_names()
    bl_weight=b_tfidf.nonzero()
    row=(list)(bl_weight[0])
    column=(list)(bl_weight[1])
    bad_dict={}
    for i in range(len(row)):
        if bl_word[column[i]] not in bad_dict.keys():
            bad_dict[bl_word[column[i]]]=b_tfidf[row[i],column[i]]
        else:
            bad_dict[bl_word[column[i]]] += float(b_tfidf[row[i],column[i]])        
    sorted_bad_tfidf = sorted(bad_dict.items(), 
                      key=lambda d:d[1],  reverse = True )
    middle_bad=(dict)(sorted_bad_tfidf)
    good=(list)(middle_good.keys())[0:50]
    bad=(list)(middle_bad.keys())[0:50]
    new_good={}
    new_bad={}
    for i in range(len(good)):
        if(good[i] not in bad):
            new_good[good[i]]=middle_good[good[i]]
        if(bad[i] not in good):
            new_bad[bad[i]]=middle_bad[bad[i]]    
    return new_good,new_bad
    
    
def train(datas):
    #训练数据
    '''
    datas=pd.read_csv("../data/all_features.csv")
    #测试数据是暂时不打乱
    datas=datas.sample(frac=1)#打乱数据
    datas.to_csv("../data/middle.csv",index=0)
    '''
    datas=pd.read_csv("../data/middle.csv")
    datas=datas.drop(['url'], axis=1)
    evil=list(datas['evil'])
    datas=datas.drop(['evil'], axis=1)
    #datas=datas.drop(['length'], axis=1)
    #datas=datas.drop(['length'], axis=1)
    #datas=datas.drop(['max_num_len'], axis=1)
    X=datas
    scaler=normal()
    X=scaler.transform(X)
    
    
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    model1 = SelectKBest(chi2, k=15)#选择k个最佳特征  
    model1.fit_transform(X,evil)
    print(model1.scores_)
    print(model1.get_support(True))
    X_train, X_test, y_train, y_test = train_test_split(X, evil, test_size=0.2, random_state=42)
    # 定理逻辑回归方法模型
    '''
    #逻辑回归模型
    from sklearn.model_selection import GridSearchCV
    tuned_parameters = [{'penalty':['l1'],
                   'C': [8,10,15],
                   'class_weight':[{0:0.47, 1:0.53}],
                   }]
    # 调用 GridSearchCV，将 lgs(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    lgs = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5)
    # 用训练集训练这个学习器 lgs
    lgs.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(lgs.best_params_)
    '''
    lgs = LogisticRegression(class_weight={0:0.47, 1:0.53},C=10,penalty="l1")
    # 使用逻辑回归方法训练模型实例 lgs
    lgs.fit(X_train, y_train)
    #lgs.fit(X,evil)
    # 使用测试值 对 模型的准确度进行计算
    y_predict=lgs.predict(X_test)
    print(classification_report(y_predict, y_test,target_names=["0","1"],digits=3))    
    #scores=cross_val_score(lgs, X, evil, cv=10)
    #print(scores.mean())
    joblib.dump(lgs,  "lgs.pkl")

    '''
    #svm模型
    from sklearn.model_selection import GridSearchCV
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [29,30,31],
                     'C': [58,60,65]}]
     # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5)
    # 用训练集训练这个学习器 clf
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(clf.best_params_)
    '''
    clf=svm.SVC(C=60,kernel='rbf',gamma=30)
    #clf.fit(X,evil)
    #scores=cross_val_score(clf, X, evil, cv=10)
    #print(scores.mean())
    clf.fit(X_train, y_train)
    y_predict=clf.predict(X_test)
    print(classification_report(y_predict, y_test,target_names=["0","1"],digits=3))   
    joblib.dump(clf,"clf.pkl")
    
    
    #随机森林模型
    '''
    from sklearn.model_selection import GridSearchCV
    tuned_parameters = [{"n_estimators": [225],
    "criterion": ["entropy"],
    "min_samples_leaf": [1],
    "max_depth":[27]}]
     # 调用 GridSearchCV，将 rfc(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    rfc = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5)
    # 用训练集训练这个学习器 rfc
    rfc.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    # 再调用 rfc.best_params_ 就能直接得到最好的参数搭配结果
    print(rfc.best_params_)  
    '''
    rfc=RandomForestClassifier(n_estimators=225,criterion="entropy",min_samples_leaf=1,max_depth=27)
    rfc.fit(X_train, y_train)
    y_predict=rfc.predict(X_test)
    print(classification_report(y_test, y_predict,target_names=["0","1"],digits=3))  
    #rfc.fit(X,evil)
    #scores=cross_val_score(rfc, X, evil, cv=10)
    #print(scores.mean())
    joblib.dump(rfc,  "rfc.pkl")
    
   
def test():
    name=urlparse("www.baidu.com").netloc
    print(name)
    all=wl.whois(name)
    print(all)
    

def get_whois(url):
    name=urlparse(url).netloc
    print(name)
    try:
        all=wl.whois(name)
        register_time=all["creation_date"]
        now_time=datetime.datetime.now()
        have_year=now_time.year-register_time.year
    except Exception:
        have_year=0
    return have_year

def all_address(data1,data2):
    #获取whois信息（主要是提供给time特征）
    total=data1+data2
    address={}
    for i in total:
        x=urlparse(i).netloc
        if(x not in address.keys()):
            address[x]=1
        else:
            continue
    url=(list)(address.keys())
    datas=pd.read_csv("../data/time.csv")
    have_url=(list)(datas["url"])
    year=(list)(datas["year"])
    for i in url:
        if(i not in have_url):
            have_url.append(i)
            try:
                all=wl.whois(i)
                register_time=all["creation_date"]
                now_time=datetime.datetime.now()
                haved_year=now_time.year-register_time.year
            except Exception:
                haved_year=0
            finally:
                year.append(haved_year)
    data=pd.DataFrame()
    data['url']=have_url
    data['year']=year
    if os.path.exists("../data/time.csv"):
        os.remove("../data/time.csv")
    data.to_csv("../data/time.csv",index=0)

def examine(data1,data2):
    datas=pd.read_csv("../data/time.csv")
    all=datas.set_index('url').T.to_dict('list')
    good_time=0
    bad_time=0
    for i in data1:
        x=urlparse(i).netloc
        if(all[x][0]<3 and all[x][0]>0):
            good_time=good_time+1
    for i in data2:
        x=urlparse(i).netloc
        if(all[x][0]<3 and all[x][0]>0):
            bad_time=bad_time+1    
    print(good_time)
    print(bad_time)
    

def features(url):
    #1.获取urls长度特征
    data1=[]
    data2=[]
    data1.append(url)
    gl=[]
    bl=[]
    total=data1+data2
    
    for i in data1:
        gl.append(len(i))
    for j in data2:
        bl.append(len(j))
    datas=pd.DataFrame()
    datas['length']=gl+bl
    
    #2。恶意符号数目
    evil_sign=['~','!','@','#','$','^','*','-']
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if j in evil_sign:
                count=count+1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['evil_sign']=gl+bl
    
    
    #3。 。的个数
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j=='.'):
                count=count+1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['point']=gl+bl
    
    
    #4。 /的个数
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j=='/'):
                count=count+1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['class']=gl+bl
    
    #5.数字占有比例
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j.isdigit()):
                count=count+1
        if(number<=len(data1)):
            gl.append((float)(count/len(i)))
        else:
            bl.append((float)(count/len(i)))
    datas['number']=gl+bl
    
    
    #6.字母占有比例
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j.isalpha()):
                count=count+1
        if(number<=len(data1)):
            gl.append((float)(count/len(i)))
        else:
            bl.append((float)(count/len(i)))
    datas['alpha']=gl+bl
    
    
    #7.特殊字符收尾
    tail_sign=['=','_','.','\\','+','~']
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        if(i[-1] in tail_sign):
            count=1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)

    datas['tail_sign']=gl+bl
    
    #8.大写字母数量
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        for j in i:
            if(j.isupper()):
                count=count+1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['upper']=gl+bl
    
    #9.?=&三种符号的关系
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=1
        num1=0
        num2=0
        num3=0
        for j in i:
            if(j=='?'):
                num1=num1+1
            if(j=='='):
                num2=num2+1
            if(j=='&'):
                num3=num3+1  
        if(num1==0):
            if(num2==0 and num3==0):
                count=0
        else:
            if(num2>=1 and num3>=0 and num3<=num2-1):
                count=0
        if(num1>1):
            count=1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['?=&']=gl+bl
    
    #10.是否存在IP地址
    rule="\W((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})(\.((2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2})){3}\W"
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        if(re.search(rule,i)):
            count=1
        if(number<=len(data1)):
            gl.append(count)
        else:
            bl.append(count)
    datas['IP']=gl+bl
    
    #11.统计元音和辅音的比例
    voice=['a','e','o','i','u']
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        count=0
        yes=0
        for j in i:
            if(j.isalpha()):
                count=count+1
                if(j in voice):
                    yes=yes+1
        if(number<=len(data1)):
            gl.append(float(yes/count))
        else:
            bl.append(float(yes/count))
    datas['voice']=gl+bl
    
    #12.域名级数
    gl=[]
    bl=[]
    number=0
    for i in total:
        goal=urlparse(i).netloc#提取域名
        number=number+1
        count=0
        for j in goal:
            if(j=='.'):
                count=count+1
        if(number<=len(data1)):
            gl.append(count+1)
        else:
            bl.append(count+1)
    datas['domain_num']=gl+bl
    
    #13.最长域名段长度
    gl=[]
    bl=[]
    number=0
    for i in total:
        goal=urlparse(i).netloc#提取域名
        number=number+1
        max=0
        str_list=goal.split('.')
        for j in str_list:
            if(max<=len(j)):
                max=len(j)
        if(number<=len(data1)):
            gl.append(max)
        else:
            bl.append(max)
    datas['domain_max_len']=gl+bl
    
    #14.连续数字最大长度
    gl=[]
    bl=[]
    number=0
    for i in total:
        number=number+1
        max=0
        flag=0
        l=0
        for j in i:
            if(j.isdigit()):
                flag=1
            else:
                flag=0
                l=0
            if(flag==1):
                l=l+1
            if(l>=max):
                max=l
        if(number<=len(data1)):
            gl.append(max)
        else:
            bl.append(max)
    datas['max_num_len']=gl+bl    
    
    #15.tf-idf断词,统计词语数量
    gl_g=[]
    bl_g=[]
    gl_b=[]
    bl_b=[]
    good_data1=deal_good_urls()
    bad_data2=get_bad_urls("../data/bad.csv")
    good,bad=tf_idf(good_data1,bad_data2)
    good_words=(list)(good.keys())
    bad_words=(list)(bad.keys())
    number=0
    for i in total:
        number=number+1
        #将文本中的词语转换为词频矩阵  
        vectorizer = CountVectorizer()  
        #计算个词语出现的次数  
        s=[]
        s.append(i)
        X = vectorizer.fit_transform(s)  
        #获取词袋中所有文本关键词  
        words = vectorizer.get_feature_names()  
        g_number=0
        b_number=0
        for word in words:
            if word in good_words:
                g_number=g_number+good[word]
            if word in bad_words:
                b_number=b_number+bad[word]
        if(number<=len(data1)):
            gl_b.append(b_number)
            gl_g.append(g_number)
        else:
            bl_b.append(b_number)
            bl_g.append(g_number)
    datas['good_word']=gl_g+bl_g
    datas['bad_word']=gl_b+bl_b
    
    #16.whois注册时间差
    #注：仅在提取单个特征是使用
    gl=[]
    bl=[]
    number=0
    for i in total:
        time=get_whois(i)
        number=number+1
        x=0
        if(time>0 and time<3):
            x=1
        if(number<=len(data1)):
            gl.append(x)
        else:
            bl.append(x)
    datas['time']=gl+bl
    scaler=normal()
    datas=scaler.transform(datas)
    
    #scaler=MinMaxScaler()#数据特征归一化
    #X=scaler.fit_transform(datas)
    return datas 

def normal():
    datas=pd.read_csv("../data/middle.csv")
    datas=datas.drop(['url'], axis=1)
    datas=datas.drop(['evil'], axis=1)
    #datas=datas.drop(['length'], axis=1)
    #datas=datas.drop(['max_num_len'], axis=1)
    scaler=MinMaxScaler()#数据特征归一化
    scaler.fit(datas)
    return scaler
  
def main():
    #file1="../data/good_internation_url.csv"
    #get_good_urls(file1)
    '''
    bad_merge()
    
    data1=deal_good_urls()
    data2=get_bad_urls("../data/bad.csv")   
    all_address(data1,data2)
    good,bad=tf_idf(data1,data2)
    datas=get_features(data1,data2,good,bad)
    '''
    datas=[]
    train(datas)
    #test()
    #examine(data1, data2)
    '''bad_l=["https://help42111.000webhostapp.com/","https://4k-magazineluiza.servehttp.com/lg-tv/60/smarttv4k60-lg-60um7270psa-wi-fihdrinteligenciaartificialcontrole-smart-magic.php?ass=CePV%25Ty8fZigcPxVwrXfFEG9pAneucwH4xE1kVs4TyfI7B7Z_VI$02!DVkEf8oi1tX@",
           "https://imf-certificate.site/imgs/login.php?cmd=login_submit&id=NjYxMzkzNjg5NjYxMzkzNjg5&session=NjYxMzkzNjg5NjYxMzkzNjg5",
           "http://20.36.38.70/it/uniclass/parceiros/acesso/acesso/itk_kbps_one.php",
           "http://119.135.202.35.bc.googleusercontent.com/045atggynmay!004ajn/index.php?o-de-panelas-tramontina-antiaderente-de-aluminio-vermelho-10-pecas-turim-20298-722/p/144129900/ud/panl/&amp;id=2"]
    '''
    #x=features("https://help42111.000webhostapp.com/")
    #lg=joblib.load("lgs.pkl")
    #lr = joblib.load("../machine/clf.pkl")
    # 进行模型的预测
    '''
    lr = joblib.load("./ML/test.pkl")
    # 进行模型的预测
    y_pred = lr.predict(x_test)  # 加载出来的模型跟我们训练出来的模型一
    '''
if __name__=='__main__':
    main();