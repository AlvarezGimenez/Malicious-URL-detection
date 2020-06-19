from flask import Flask,request,redirect
from flask_bootstrap import Bootstrap
from flask import render_template,url_for
from machine.deal import features
from sklearn.externals import joblib
import urllib.request
import requests
from bs4 import BeautifulSoup
import whois as wl
import json
import xml.etree.ElementTree as ET
import socket
from urllib.parse import urlparse
import geoip2.database
from keras.models import load_model
from deeplearn.deal import get_pre_data
import keras
import pyecharts.options as opts
from pyecharts.charts import Gauge
from snapshot_selenium import snapshot
from pyecharts.render import make_snapshot
from pyecharts.options.global_options import InitOpts

app=Flask(__name__,static_folder="static",template_folder="templates")
bootstrap=Bootstrap(app)

def model(url):
    lg=joblib.load("../machine/rfc.pkl")
    x=features(url)
    result=lg.predict(x)
    print(result)
    if(result[0]==0):
        return 0
    else:
        return 1
def LSTM(url):
    ls =load_model('../deeplearn/my_model.h5')
    data=get_pre_data(url)
    results=ls.predict(data)
    y_p=0
    grade=(float)(results[0][0])
    if(results[0]>=0.5):
        y_p=1
    else:
        y_p=0
    return y_p,round(grade,2)

def draw(grade):
    grade=(int)((grade)*100)
    c= (
    Gauge(init_opts=opts.InitOpts(width="450px", height="250px"))
    .add("恶意可能性评估", [("", grade)],axisline_opts=opts.AxisLineOpts(
            linestyle_opts=opts.LineStyleOpts(
                color=[(0.2, "#00FF00"),(0.4, "#009900"), (0.6, "#FFFF00"), (0.8, "#FF6600"),(1, "#FF0000")], width=30
                )
            ),
         )
    .render("static/test.html")
    )


def get_information(url):
    #phishtank key:822c4203632b270307eb0f82e381293d16357f4cdc5b1efb3f33c8533b72c682
    judge_url="https://checkurl.phishtank.com/checkurl/index.php?url="+url
    req=urllib.request.Request(judge_url)
    resp=urllib.request.urlopen(req)
    data=resp.read().decode('utf-8')
    root=ET.fromstring(data)
    try:
        result_node=root.find("results")
        mid_node=result_node.find("url0")
        node=mid_node.find("verified")
        if(node.text=="true"):
            result=1
    except Exception:
        result=0
    return result


def get_ip(url):
    try:
        domain=urlparse(url).netloc
        address=socket.getaddrinfo(domain,None)[-1][4][0]
        print(socket.getaddrinfo(domain,None))
    except Exception:
        address="None"
    reader=None
    response=None
    try:
        reader = geoip2.database.Reader('../GeoLite2-City.mmdb')
        response = reader.city(address)
        country=response.country.names["zh-CN"]   
    except Exception:
        country="None"
    try:
        city=response.city.name        
    except Exception:
        city="None"
    return address,country,city

@app.route("/",methods=["GET","POST"])
def test():
    if request.method == "POST":
        print("ok")
        data = request.get_json()
        url=data["url"]
        keras.backend.clear_session()
        evil,grade=LSTM(url)
        #phish=get_information(url)
        ip,country,city=get_ip(url)
        datas={}
        datas["evil"]=evil
        datas["url"]=url
        #datas["phish"]=phish
        datas["ip"]=ip
        datas["area"]=str(country)
        datas["grade"]=grade
        draw(grade)
        return json.dumps(datas)
    else:
        return render_template("url.html")    

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'),404

@app.errorhandler(500)
def not_response(error):
    return render_template('404.html'),500

'''
def main():
    draw(95)
if __name__ == '__main__':
    main()

'''