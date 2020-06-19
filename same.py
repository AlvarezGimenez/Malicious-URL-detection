import xlrd
import xlwt
import pandas as pd
import pdfplumber
def get_name():
    readbook = xlrd.open_workbook(r'data/all_grade.xls')
    worksheet=readbook.sheet_by_index(0)
    name=worksheet.col_values(1)
    _class=worksheet.col_values(2)
    final=dict(zip(name[3:],_class[3:]))
    return final

def find_name(all_data):
    data1=pd.read_csv("data/grade.csv")
    data2=pd.read_csv("data/grade_1.csv")
    data3=pd.read_csv("data/grade_2.csv")
    data_name=list(data1["姓名"])+list(data2["姓名"])+list(data3["姓名"])
    data_grade=list(data1["成绩"])+list(data2["成绩"])+list(data3["成绩"])
    data=dict(zip(data_name,data_grade))
    names=(list)(all_data.keys())
    all_names=(list)(data_name)
    final=0
    good=0
    for name in all_names:
        if(name in names):
            final=final+1
            if(data[name]>300):
                good=good+1
            print(name,data[name],all_data[name])
    print(final)
    print(good)
    print((float)((good-1)/final))        
        
def txt(all_name):
    datas = []
    final=0
    names=(list)(all_name.keys())
    for line in open("data/total.txt","r",encoding="utf-8"): #设置文件对象并读取每一行文件
        l=line.split();  
        datas.append(l[4])
    for data in datas:
        if(data in names):
            print(data)
            final=final+1
    print(final)
        
def main():
    all_data=get_name()
    #find_name(all_data)
    txt(all_data)

if __name__ == '__main__':
    main()