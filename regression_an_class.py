#%%
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pandas as pd
from icecream import ic
#%%
'''
상속이란 무엇인가?
1. 부모 class
2. 자식 class
'''

'''
1. 부모 class: tf.keras.Model
2. 자식 class: MNISTClassifier
'''

'''
class MNISTClassifier(tf.keras.Model):
    def __init__(self):
        
        super: 자식 class에 부모 class가 가진 method들을 initialize함
        
        super(MNISTClassifier, self).__init__()
        
        
        우리가 원하는 자식 class의 변수들을 저장
        
        self.dense = layers.Dense(10)
        
    
    def: method를 정의
    만약, 이름이 겹치는 method가 있을 경우에는?
    덮어쓰기!
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x) 

객체(instance)를 정의
classifier = MNISTClassifier() 

classifier.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Fit model
classifier.fit(x_train, y_train, epochs = 10)
'''
#%%

from tensorflow.keras.utils import to_categorical

class OrdinalRegression():
    def __init__(self, k, p, lr):
        # super(OrdianlRegression, self).__init__()
        self.k = k
        self.p = p
        self.lr = lr
        self.theta = tf.Variable(tf.random.normal(shape=(1, self.k-1), stddev=0.1), trainable=True, dtype=tf.float32)
        self.beta = tf.Variable(tf.random.normal(shape=(self.p, 1), stddev=0.1), trainable=True, dtype=tf.float32)
    
    def loss_function(self, alpha, x, y):
        n = x.shape[0]
        y_indicator = to_categorical(y, num_classes=self.k)
        
        reg = tf.matmul(x, self.beta)
        mat = alpha + reg
        
        left = tf.concat((tf.nn.sigmoid(mat), tf.ones((n, 1))), axis=1)  # axis=0이면 밑으로 붙음
        right = tf.concat((tf.zeros((n, 1)), tf.nn.sigmoid(mat)), axis=1)

        loss = - tf.reduce_sum(y_indicator * tf.math.log(left - right + 1e-8)) / n
        return loss        

    def train_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.theta)
            tape.watch(self.beta)
            
            alpha = [self.theta[0, 0]]
            for i in range(1, self.theta.shape[1]):
                alpha.append(alpha[i-1] + tf.math.square(self.theta[0, i]))  
            alpha = tf.stack(alpha)[tf.newaxis, :]                      

            loss = self.loss_function(alpha, x, y)
        
        grad_theta = tape.gradient(loss, self.theta) #앞엔 target이므로 보통 loss. 뒤엔 학습시키려는 variable tensor
        grad_beta = tape.gradient(loss, self.beta)
        self.theta = self.theta - self.lr * grad_theta
        self.beta = self.beta - self.lr * grad_beta
        return loss, self.theta, self.beta
    
    def accuracy(self, y):
        n = y.shape[0]
        alpha = [self.theta[0, 0]]
        for i in range(1, self.theta.shape[1]):
            alpha.append(alpha[i-1] + tf.math.square(self.theta[0, i]))
        alpha = tf.stack(alpha)[tf.newaxis, :]
        reg = tf.matmul(x, self.beta)
        mat = alpha + reg

        left = tf.concat((tf.nn.sigmoid(mat), tf.ones((n, 1))), axis=1)
        right = tf.concat((tf.zeros((n, 1)), tf.nn.sigmoid(mat)), axis=1)
        y_pred = tf.argmax(left - right, axis=1)

        table = pd.crosstab(y, y_pred.numpy(),rownames=['True'], colnames=['Predicted'], margins=True)
        acc = np.sum(np.diag(table)[:-1]) / n
        return table, acc
#%%
boston = pd.read_csv('bostonhousing_ord.csv', encoding='cp949')
boston.head()
x = np.array(boston.iloc[:,1:])
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0) # x를 표준화
x = tf.cast(x, tf.float32)                       # tf에 알맞는 형태로 캐스팅
n, p = x.shape
#%%
y = np.array(boston.iloc[:,0])
y = y - 1 # broadcasting
k = len(np.unique(y))
#%%
ordinalregression = OrdinalRegression(k = k, p = p, lr = 0.01)
#%%
epochs = 1000
for step in range(epochs):
    loss, theta, beta = ordinalregression.train_step(x, y)
    if step % 10 == 0:
        ic(step, loss)
#%%
table, acc = ordinalregression.accuracy(y)
ic(table)
ic(acc)
#%%