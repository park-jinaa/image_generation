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