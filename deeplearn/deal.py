import numpy as np
import pandas as pd
import machine.deal as dl
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab 
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense,LSTM,SimpleRNN,Dropout
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import load_model
from setuptools.dist import sequence
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias', 
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        x = K.permute_dimensions(inputs, (0, 2, 1))
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
    
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
 
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def get_data():
    data1=dl.deal_good_urls()
    data2=dl.get_bad_urls("../data/bad.csv") 
    evil=[]
    l=[]
    for i in data1:
        evil.append(0)
        l.append(len(i))
    for i in data2:
        evil.append(1)
        l.append(len(i))
    data=data1+data2
    datas=pd.DataFrame()
    datas["url"]=data
    datas["evil"]=evil
    datas["length"]=l
    q1, q3 = np.percentile(l, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + (iqr * 1.5)
    print(upper_bound)
    return datas


def deal_data(datas):
    maxlen = 24  # 句子最长词语数量
    max_features = 20000 #按词频大小取样本前20000个词
    input_dim = max_features #词库大小 必须>=max_features
    #batch_size = 128 #batch数量
    #output_dim = 64 #词向量维度
    #epochs = 8 #训练批次
    batch_size =128 #batch数量
    output_dim =128 #词向量维度
    epochs=10

    samples=datas["url"]
    tokenizer=Tokenizer(num_words=None)#只考虑前一千个最常见单词，设成你想要的维度
    tokenizer.fit_on_texts(samples)#构建单词索引
    
    """分词器（tokenizer）支持多种模式的向量化模型"""
    #1.将字符串转化为整数索引组成的列表：[[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]
    sequences=tokenizer.texts_to_sequences(samples)
    l=[]
    for senquence in sequences:
        l.append(len(senquence))
    q1, q3 = np.percentile(l, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + (iqr * 1.5)
    print(upper_bound)
    data = pad_sequences(sequences, maxlen=maxlen)
    labels = np.asarray(datas["evil"])
    indices = np.arange(data.shape[0])
    #打乱顺序
    np.random.shuffle(indices)
    data = data[indices]
    data[data>=20000]=0
    labels = labels[indices]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = Sequential() 
    model.add(Embedding(input_dim, output_dim, input_length=maxlen)) #词嵌入:词库大小、词向量维度、固定序列长度
    model.add(Dropout(0.5))
    #构建LSTM层
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
    #model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5))
    model.add(AttentionLayer())
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    #RMSprop优化器 二元交叉熵损失
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) #训练
    history = LossHistory()
    model.fit(X_train, y_train, batch_size, epochs,shuffle=True,validation_data=(X_test,y_test),callbacks=[history]) #模型可视化
    #model.summary()
    scores =model.evaluate(X_test, y_test, batch_size=200, verbose=0)
    print(scores)
    history.loss_plot('epoch')
    #model.save("my_model.h5")
    #ls =load_model('my_model.h5')
    #ls.predict(X_test[-1,:])
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
def get_pre_data(url):
    maxlen = 128
    datas=get_data()
    datas=datas.append([{'url':url}], ignore_index=True)
    samples=datas["url"]
    tokenizer=Tokenizer(num_words=None)#只考虑前一千个最常见单词，设成你想要的维度
    tokenizer.fit_on_texts(samples)#构建单词索引
    
    """分词器（tokenizer）支持多种模式的向量化模型"""
    #1.将字符串转化为整数索引组成的列表：[[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]
    sequences=tokenizer.texts_to_sequences(samples)
    data = pad_sequences(sequences, maxlen=maxlen)
    data[data>=20000]=0
    return data[-1:,]


def main():
    datas=get_data()
    datas = datas.sample(len(datas))
    deal_data(datas)
    #get_pre_data("https://www.baidu.com")
if __name__=='__main__':
    main(); 