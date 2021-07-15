#%%
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
#%%

mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test)=mnist.load_data()

# 데이터 탐색
import matplotlib.pyplot as plt
plt.imshow(x_train[0],cmap='gray')
y_train[0]
x_train[0]


# 데이터 전처리
from tensorflow.keras.utils import to_categorical
def percentage_onehot(x,y):
    
    # float32로 변경, 학습성능 향상을 위해 픽셀값을 0~1 사이 값으로 변환
    x=np.array(x/255.0, dtype=np.float32)  
    y=np.array(y, dtype=np.float32)    
    
    # 원핫인코딩 (for categorical_crossentropy)
    OH_y = to_categorical(y)
    return x, OH_y

x_train, y_train = percentage_onehot(x_train, y_train)
x_test, y_test = percentage_onehot(x_test, y_test)
#%%
x_train.shape[1]



#%%

'''
from tensorflow.keras.layers import Layer, Input, Dense, Flatten # 수정
from tensorflow.keras.models import Model
import tensorflow as tf


input_size_width = x_train.shape[1]
input_size_height = x_train.shape[2]

def create_model():
  input_=Input(shape=(input_size_width,input_size_height))
  x = Flatten()(input_)
  x = Dense(100, activation='relu')(x)
  x = Dense(30, activation='relu')(x)
  output_ = Dense(10, activation='softmax')(x)
'''
from tensorflow.keras import layers 
from tensorflow.keras.models import Model
import tensorflow as tf

# functional_api로 모델 생성
input_size_width = x_train.shape[1]
input_size_height = x_train.shape[2]

def create_model():
  input_=layers.Input(shape=(input_size_width,input_size_height))
  x = layers.Flatten()(input_)
  x = layers.Dense(100, activation='relu')(x)
  x = layers.Dense(30, activation='relu')(x)
  output_ = layers.Dense(10, activation='softmax')(x)

  model=Model(inputs=input_, outputs=output_)  
  
  return model

model = create_model()
model.summary()  # None: keras framework 내에서 2차원 데이터 형태를 layers.Input인자에 넣으면 
                 #       fit할 땐 batch까지 3차원으로 들어올거라고 인식해서 우선 None 처리

#%%
#모델 loss와 optimizer 설정하고 학습
from tensorflow.keras.optimizers import Adam
'''
# from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.metrics import Accuracy
'''
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy']) 


# 학습
history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=30, validation_split=0.15) 


#학습 이력
print(history.history['loss'])  
print(history.history['accuracy'])

# 테스트 데이터 세트로 모델 성능 검증
model.evaluate(x_test, y_test, batch_size=64, verbose=1)


#%%
# ReduceLROnPlateau를 이용한 learning rate 조정 - 성능 향상이 없으면 Learning rate를 동적으로 감소
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduceLR = ReduceLROnPlateau(
    monitor='val_loss',  # val_손실 기준으로 callback 호출
    factor=0.5,          # callback 호출시 학습률을 1/2로 줄임
    patience=5,          # moniter값의 개선없을 시 5번을 참아
    mode='min',          # moniter가 손실이니까 min
    min_lr=1e-5,
    min_delta=0.01,      # 개선된 것으로 간주한 최소한의 변화량
    cooldown=2,          # 쿨타임
    verbose=1
)
model = create_model()

# 정리
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=50, validation_split=0.15, callbacks=[reduceLR])

model.evaluate(x_test, y_test, batch_size=64)



#LearningRateScheduler를 이용한 learning rate 조정
def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

model = create_model()
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history= model.fit(x=x_train, y=y_train, 
                   batch_size = 128, 
                   epochs = 50, 
                   validation_split = 0.15,
                   callbacks = [callback],
                   verbose = 1)

model.evaluate(x_test, y_test, batch_size=64)










