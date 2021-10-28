#%%
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pandas as pd
from icecream import ic
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
from tensorflow.keras.utils import to_categorical

class LinearModel(K.models.Model):
    def __init__(self, k, p):
        super(LinearModel, self).__init__()
        self.k = k
        self.p = p
        self.theta = tf.Variable(tf.random.normal(shape=(1, self.k-1), stddev=0.1), trainable=True, dtype=tf.float32)
        self.beta = tf.Variable(tf.random.normal(shape=(self.p, 1), stddev=0.1), trainable=True, dtype=tf.float32)

    def call(self, x):
        n = x.shape[0]

        # reparametrization
        alpha = [self.theta[0, 0]]
        for i in range(1, self.theta.shape[1]):
            alpha.append(alpha[i-1] + tf.math.square(self.theta[0, i]))  
        alpha = tf.stack(alpha)[tf.newaxis, :]                      

        reg = tf.matmul(x, self.beta)
        mat = alpha + reg
        
        left = tf.concat((tf.nn.sigmoid(mat), tf.ones((n, 1))), axis=1)  # axis=0이면 밑으로 붙음
        right = tf.concat((tf.zeros((n, 1)), tf.nn.sigmoid(mat)), axis=1)

        return left - right
    
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
linear_model = LinearModel(k, p)
optimizer = K.optimizers.SGD(lr=0.1)
y_indicator = to_categorical(y, num_classes=k)
#%%
@tf.function
def loss_function(prob_diff, y_indicator):
    loss = - tf.reduce_mean(y_indicator * tf.math.log(prob_diff + 1e-8))
    return loss
#%%
@tf.function
def train_step(x, y_indicator):
    with tf.GradientTape() as tape:
        prob_diff = linear_model(x)

        loss = loss_function(prob_diff, y_indicator)

    grad = tape.gradient(loss, linear_model.trainable_weights)
    optimizer.apply_gradients(zip(grad, linear_model.trainable_weights))
    return loss
#%%
epochs = 1000
for step in range(epochs):
    loss = train_step(x, y_indicator)
    if step % 10 == 0:
        ic(step, loss)
#%%
table, acc = linear_model.accuracy(y)
ic(table)
ic(acc)
#%%