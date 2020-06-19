import pandas as pd
import numpy as py
import csv

def bad_merge():
    #每次合并更新的恶意urls数据，在add_name处输入文件名，即可合并
    main_name='../data/bad.csv'
    add_name='../data/316.csv'
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
    
