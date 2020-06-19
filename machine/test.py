import numpy as np
import pandas as pd
import deal
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab 
def get_data():
    data1=deal.deal_good_urls()
    data2=deal.get_bad_urls("../data/bad.csv") 
    evil=[]
    for i in data1:
        evil.append(0)
    for i in data2:
        evil.append(1)
    data=data1+data2
    datas=pd.DataFrame()
    datas["url"]=data
    datas["evil"]=evil
    return datas

def m():
    samples = ['https://passport.baidu.com/v2/?login&tpl=mn&u=http%3A%2F%2Fwww.baidu.com%2F', 'https://www.baidu.com/',"http://au-ok.com"]
    tokenizer=Tokenizer(num_words=1000)#只考虑前一千个最常见单词，设成你想要的维度
    tokenizer.fit_on_texts(samples)#构建单词索引
 
    """分词器（tokenizer）支持多种模式的向量化模型"""
    #1.将字符串转化为整数索引组成的列表：[[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]
    sequences=tokenizer.texts_to_sequences(samples)
 
    #2.可以得到onehot二进制表示的列表：[[ 0.  1.  1. ...,  0.  0.  0.][ 0.  1.  0. ...,  0.  0.  0.]]
    one_hot_results=tokenizer.texts_to_matrix(samples,mode='binary')
    print(sequences)
    print(one_hot_results)

def deal_data(datas):
    maxlen = 100  # 句子最长词语数量
    max_features = 20000 #按词频大小取样本前20000个词
    input_dim = max_features #词库大小 必须>=max_features
    batch_size = 128 #batch数量
    output_dim = 40 #词向量维度
    epochs = 2 #训练批次
    units = 32 #RNN神经元数量

    samples=datas["url"]
    tokenizer=Tokenizer(num_words=None)#只考虑前一千个最常见单词，设成你想要的维度
    tokenizer.fit_on_texts(samples)#构建单词索引
    
    """分词器（tokenizer）支持多种模式的向量化模型"""
    #1.将字符串转化为整数索引组成的列表：[[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]
    sequences=tokenizer.texts_to_sequences(samples)
    data = pad_sequences(sequences, maxlen=maxlen)
    labels = np.asarray(datas["evil"])
    indices = np.arange(data.shape[0])
    #打乱顺序
    np.random.shuffle(indices)
    data = data[indices]
    data[data>=20000]=0
    labels = labels[indices]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    print(y_train)
    from keras.models import Sequential
    from keras.layers import Embedding, Flatten, Dense,LSTM,SimpleRNN
    model = Sequential() #词嵌入:词库大小、词向量维度、固定序列长度
    model.add(Embedding(input_dim, output_dim, input_length=maxlen)) #平坦化: maxlen*output_dim
    #model.add(Flatten()) #输出层: 2分类
    #RNN Cell
    model.add(SimpleRNN(units, return_sequences=True)) #返回序列全部结果
    model.add(SimpleRNN(units, return_sequences=False)) #返回序列最尾结果
    #构建LSTM层
    #model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    #RMSprop优化器 二元交叉熵损失
    
    model.compile('rmsprop', 'binary_crossentropy', ['acc']) #训练
    model.fit(X_train, y_train, batch_size, epochs,validation_data=(X_test,y_test)) #模型可视化
    model.summary()
    '''
    y_pre=model.predict(X_test)
    y_p=[]
    for i in y_pre:
        if(i[0]>=0.5):
            y_p.append(1)
        else:
            y_p.append(0)
    print(len(y_p))
    print(len(X_test))
    gl=[]
    for i in range(1,len(X_test)+1):
        gl.append(i)
    y=[]
    count=0
    for i in range(0,len(y_p)):
        if(y_p[i]==y_test[i]):
            y.append(0)
        else:
            y.append(1)
            count=count+1
    plt.scatter(gl,y,color="green")
    plt.show()
    total=len(y_p)
    print((total-count)/total)
    '''
    '''
    #-----------------------------------建模与训练-----------------------------------
    #激活神经网络
    model.compile(optimizer = 'rmsprop', #RMSprop优化器
                 loss = 'binary_crossentropy', #二元交叉熵损失
                 metrics = ['accuracy'] #计算误差或准确率
    ) #训练
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                         verbose=2,
                         validation_split=.1 #取10%样本作验证
                         ) 
    #-----------------------------------预测与可视化-----------------------------------
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(range(epochs), accuracy)
    plt.plot(range(epochs), val_accuracy)
    plt.show()
    '''
def main():
    datas=get_data()
    datas = datas.sample(len(datas))
    deal_data(datas)
if __name__=='__main__':
    main(); 