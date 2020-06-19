import pandas as pd


def main():
    datas=pd.read_csv("data/time_test.csv")
    print(datas)
    all=(list)(datas[",url,year"])
    url=[]
    year=[]
    for i in all:
        x=i.split(",")
        url.append(x[1])
        year.append(x[2])
    data=pd.DataFrame()
    data["url"]=url
    data["year"]=year
    data.to_csv("data/time.csv",index=0)

if __name__ == '__main__':
    main()