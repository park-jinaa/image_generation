#%%
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as K
import numpy as np
import pandas as pd
from icecream import ic
#%%
boston = pd.read_csv('bostonhousing_ord.csv', encoding='cp949')
boston.head()
x = np.array(boston.iloc[:,1:])
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0) # x를 표준화
n, p = x.shape
#%%
y = np.array(boston.iloc[:,0])
y = y - 1 # broadcasting
k = len(np.unique(y))
#%%
# '''linear'''
# beta = tf.Variable(tf.random.normal(shape=(p, 1), stddev=0.1), trainable=True, dtype=tf.float32)
# tf.matmul(x, beta).shape

# '''non-linear'''
# dense1 = layers.Dense(5, activation='relu')
# dense2 = layers.Dense(1, activation='linear')

# h = dense1(x)
# dense1.weights[0].shape
# h.shape
# h = dense2(h)
# dense2.weights[0].shape
# h.shape
#%%
from tensorflow.keras.utils import to_categorical

class NeuralModel(K.models.Model):
    def __init__(self, k, p):
        super(NeuralModel, self).__init__()
        self.k = k
        self.p = p
        
        # linear (모수가 데이터에 의존하지 않는다!)
        self.theta = tf.Variable(tf.random.normal(shape=(1, self.k-1), stddev=0.1), trainable=True, dtype=tf.float32)
        # self.beta = tf.Variable(tf.random.normal(shape=(self.p, 1), stddev=0.1), trainable=True, dtype=tf.float32)

        # nonlinear
        # self.theta1 = layers.Dense(7, activation='relu')
        # self.theta1 = layers.Dense(self.k-1, activation='linear')
        self.beta1 = layers.Dense(5, activation='relu')
        self.beta2 = layers.Dense(1, activation='linear')

    def call(self, x):
        n = x.shape[0]

        # reparametrization
        # theta = self.theta1(x)
        # theta = self.theta2(theta)
        # theta = tf.split(theta, num_or_size_splits=self.k-1, axis=-1)
        # alpha = [theta[0]]
        # for i in range(1, len(theta)):
        #     alpha.append(alpha[i-1] + tf.math.square(theta[i]))  
        # alpha = tf.squeeze(tf.transpose(tf.stack(alpha), perm=[1, 0, 2]), axis=-1)

        alpha = [self.theta[0, 0]]
        for i in range(1, self.theta.shape[1]):
            alpha.append(alpha[iff-1] + tf.math.square(self.theta[0, i]))  
        alpha = tf.stack(alpha)[tf.newaxis, :]      

        h = self.beta1(x)
        h = self.beta2(h)
        mat = alpha + h
        
        left = tf.concat((tf.nn.sigmoid(mat), tf.ones((n, 1))), axis=1)  # axis=0이면 밑으로 붙음
        right = tf.concat((tf.zeros((n, 1)), tf.nn.sigmoid(mat)), axis=1)

        return left - right
    
    def accuracy(self, x, y):
        n = y.shape[0]

        # theta = self.theta1(x)
        # theta = self.theta2(theta)
        # alpha = [theta[:, [0]]]
        # for i in range(1, theta.shape[1]):
        #     alpha.append(alpha[i-1] + tf.math.square(theta[:, [i]]))  
        # alpha = tf.stack(alpha)

        alpha = [self.theta[0, 0]]
        for i in range(1, self.theta.shape[1]):
            alpha.append(alpha[i-1] + tf.math.square(self.theta[0, i]))  
        alpha = tf.stack(alpha)[tf.newaxis, :]      

        h = self.beta1(x)
        h = self.beta2(h)
        mat = alpha + h

        left = tf.concat((tf.nn.sigmoid(mat), tf.ones((n, 1))), axis=1)
        right = tf.concat((tf.zeros((n, 1)), tf.nn.sigmoid(mat)), axis=1)
        y_pred = tf.argmax(left - right, axis=1)

        table = pd.crosstab(y, y_pred.numpy(),rownames=['True'], colnames=['Predicted'], margins=True)
        acc = np.sum(np.diag(table)[:-1]) / n
        return table, acc
#%%
neural_model = NeuralModel(k, p)
optimizer = K.optimizers.SGD(learning_rate=0.05)
y_indicator = to_categorical(y, num_classes=k)
#%%
@tf.function
def loss_function(prob_diff, y):
    loss = - tf.reduce_mean(y * tf.math.log(prob_diff + 1e-8))
    return loss
#%%
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        prob_diff = neural_model(x)
        loss = loss_function(prob_diff, y)

    grad = tape.gradient(loss, neural_model.trainable_weights)
    optimizer.apply_gradients(zip(grad, neural_model.trainable_weights))
    return loss
#%%
epochs = 1500
for step in range(epochs):
    loss = train_step(x, y_indicator)
    if step % 10 == 0:
        print(step, loss)
#%%
table, acc = neural_model.accuracy(x, y)
ic(table)
ic(acc)
#%%
'''
끝
'''




























'''
13차원을 1차원으로 바꾸는 과정에서
원래는 회귀분석의 x*beta를 사용했는데
nonlinear한 신경망 2개의 layer로 해봄.
또한 있던 모형을 클래스로 짜봄.


'''
#%%