import csv

def bad_merge():
    main_name="watch.csv"
    f= open("txt1.log","r")   #设置文件对象
    datas = f.readlines()
    new_line=[]
    w=["浏览器版本","","用户名","","密码","","被爆破的网页","","","","","攻击IP","","攻击时间","","","爆破次数"]
    with open(main_name,"a+",encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile,lineterminator='\n')
        writer.writerow(w)
        count=0
        for i in datas:
            try:
                count=count+1
                l=[]
                x=i.split()
                y=x[10].split('&')
                l.append(x[0])
                l.append("")
                l.append(y[0])
                l.append("")
                l.append(y[1])
                l.append("")
                l.append(x[12])
                l.append("")
                l.append("")
                l.append("")
                l.append("")
                l.append(x[13])
                l.append("")
                l.append(x[16])
                l.append("")
                l.append("")
                l.append(count)
                writer.writerow(l)
            except:
                print("")
        
def main():
    bad_merge()
if __name__=='__main__':
    main(); 
    
