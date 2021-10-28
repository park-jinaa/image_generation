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

from tensorflow.keras.utils import to_categorical
y_indicator = to_categorical(y, num_classes=k)
y_indicator.shape

#%%

'''  
1. tf.random.normal : 정규분포에서 상수 추출 = 시작값
2. tf.Variable : 정해진 상수를 변수로 변환
'''
theta = tf.Variable(tf.random.normal(shape=(1, k-1), stddev=0.1), trainable=True, dtype=tf.float32)
beta = tf.Variable(tf.random.normal(shape=(p, 1), stddev=0.1), trainable=True, dtype=tf.float32)

#%%

@tf.function # 함수를 tensorflow 함수로 변환시켜 속도를 향상시킴
def loss_function(alpha, beta, x, y_indicator):
    reg = tf.matmul(x, beta)
    mat = alpha + reg
    
    left = tf.concat((tf.nn.sigmoid(mat), tf.ones((n, 1))), axis=1)  # axis=0이면 밑으로 붙음
    right = tf.concat((tf.zeros((n, 1)), tf.nn.sigmoid(mat)), axis=1)

    loss = - tf.reduce_sum(y_indicator * tf.math.log(left - right + 1e-8)) / n
    return loss

#%%
lr = 0.01

@tf.function
def train_step(theta, beta, x, y_indicator):
    with tf.GradientTape(persistent=True) as tape:
        '''
        기록할 대상
        '''
        tape.watch(theta)
        tape.watch(beta)
        '''
        alpha 제약조건 변환
        '''
        alpha = [theta[0, 0]]
        for i in range(1, theta.shape[1]):
            alpha.append(alpha[i-1] + tf.math.square(theta[0, i]))  # 알파1+(알파2-알파1)^2
        alpha = tf.stack(alpha)[tf.newaxis, :]                      

        loss = loss_function(alpha, beta, x, y_indicator)

    grad_theta = tape.gradient(loss, theta) #앞엔 target이므로 보통 loss. 뒤엔 학습시키려는 variable tensor
    grad_beta = tape.gradient(loss, beta)
    theta = theta - lr * grad_theta
    beta = beta - lr * grad_beta
    return loss, theta, beta
#%%
epochs = 1000
for step in range(epochs):
    loss, theta, beta = train_step(theta, beta, x, y_indicator)
    if step % 10 == 0:
        ic(step, loss)
#%%
'''acc 찾기'''
alpha = [theta[0, 0]]
for i in range(1, theta.shape[1]):
    alpha.append(alpha[i-1] + tf.math.square(theta[0, i]))
alpha = tf.stack(alpha)[tf.newaxis, :]
reg = tf.matmul(x, beta)
mat = alpha + reg

left = tf.concat((tf.nn.sigmoid(mat), tf.ones((n, 1))), axis=1)
right = tf.concat((tf.zeros((n, 1)), tf.nn.sigmoid(mat)), axis=1)
y_pred = tf.argmax(left - right, axis=1)

table = pd.crosstab(y, y_pred.numpy(),rownames=['True'], colnames=['Predicted'], margins=True)
ic(table)
acc = np.sum(np.diag(table)[:-1]) / n
ic(acc)

# %%






# %%
test_list1 = [1, 2, 3]
test_list2 = [10, 20, 30]

t1 = tf.Variable(test_list1, dtype=tf.float32)
t2 = tf.Variable(test_list2, dtype=tf.float32)

with tf.GradientTape() as tape:
    t3 = tf.square(t1) * t2

gradients = tape.gradient(t3, [t1, t2]) #t3에 의해 미분된 t1, t2
print(gradients[0])
'''
tf.Tensor([20. 80. 180.], shape=(3,), dtype=float32)
'''

print(gradients[1])
'''
tf.Tensor([1. 4. 9.], shape=(3,), dtype=float32)
'''