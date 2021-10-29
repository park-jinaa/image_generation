#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers

import numpy as np
#%%
mnist = K.datasets.mnist
(x_train, y_train),(x_test, y_test)=mnist.load_data()
#%%
x_train = x_train / 255.
x_train = x_train.reshape(-1, 784)
y_train_onehot = tf.keras.utils.to_categorical(y_train)
#%%
train_datasets = tf.data.Dataset.from_tensor_slices((x_train, y_train_onehot)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(128) #학습용 batch dataset: 128개
#%%
inputs = layers.Input(shape=(x_train.shape[1]))
h = layers.Dense(100, activation='relu')(inputs)
h = layers.Dense(30, activation='relu')(h)
outputs = layers.Dense(10, activation='linear')(h)
model = K.models.Model(inputs, outputs)

model.summary()
#%%
optimizer = K.optimizers.Adam()

for _ in range(30):
    x_batch, y_batch = next(iter(train_datasets))

    with tf.GradientTape() as tape:
        pred = model(x_batch)
        loss = tf.nn.softmax_cross_entropy_with_logits(y_batch, pred)
    gradient = tape.gradient(loss, model.trainable_variables)
    print(np.sum(gradient[0].numpy()))
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
#%%

import numpy as np
import math

N=100000000
# x = np.random.normal(loc=0, scale=0.2, size=(n, 1))
z = np.random.normal(0,1,N)
w = np.exp(-0.5 * z**2) / (np.sqrt(2 * math.pi) * np.exp(-(z-4)))
np.mean(w)
# %%
import numpy as np
np.random.seed(1)
p = 2
true_beta = np.array([[1], [0.5]])
n = 1000
x = np.random.normal(loc=0, scale=0.2, size=(n, 1))
x = np.hstack((np.ones((n, 1)), x))
x[:5, :]
# 모수값을 계산하고 poisson 분포를 이용해 y를 생성
parm = np.exp(x @ true_beta)
parm[:5, :]
y = np.random.poisson(parm)
y[:5, :]


#%%
N = 100000
y = 0
for i in range(N):
    z=-np.log(np.random.uniform())+4
    if z>4:
        y+=np.exp(-1/2*z**2+z-4)/np.sqrt(2*np.pi)
    else:
        y+=0.5
estimator=y/N

import scipy.stats
real=1-scipy.stats.norm(0,1).cdf(4)
np.abs(real - estimator)

#%%