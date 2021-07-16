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
from tensorflow.keras import layers 
from tensorflow.keras.models import Model
import tensorflow as tf

# functional_api로 모델 생성
input_size_width = x_train.shape[1]
input_size_height = x_train.shape[2]

def create_model():
    input_=layers.Input(shape=(input_size_width, input_size_height))
    x = layers.Flatten()(input_)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(30, activation='relu')(x)
    output_ = layers.Dense(10, activation='softmax')(x) # 실제값 : multinomial distribution
    model=Model(inputs=input_, outputs=output_)  

    return model

'''
softmax의 주석
'''

model = create_model()
model.summary()  # None: keras framework 내에서 2차원 데이터 형태를 layers.Input인자에 넣으면 
								 #       fit할 땐 batch까지 3차원으로 들어올거라고 인식해서 우선 None 처리
#%%
# 모델 loss와 optimizer 설정 및 학습
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy']) 

# 학습
history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=30, validation_split=0.15) 

#학습 이력
print(history.history['loss'])  
print(history.history['accuracy'])

# 테스트 데이터 세트로 모델 성능 검증
model.evaluate(x_test, y_test, batch_size=64, verbose=1)
#%%
'''
GradientTape(그래디언트 테이프)
텐서플로는 자동 미분(주어진 입력 변수에 대한 연산의 그래디언트(gradient)를 계산하는 것) 을 위한 tf.GradientTape API를 제공
tf.GradientTape는 context 안에서 실행된 모든 연산을 tape에 기록
그 다음 텐서플로는 후진 방식 자동 미분(reverse mode differentiation)을 사용해 tape에 "기록된" 연산의 그래디언트를 계산
gradient 계산 후 weight update
'''

'''
LearningRateScheduler를 이용한 learning rate 조정
 - exponential
'''
def lr_scheduler(epoch, lr):
	if epoch < 5:
		return lr
	else:
		return lr * tf.math.exp(-0.1)

model = create_model()
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
history= model.fit(x=x_train, y=y_train, 
									 batch_size = 128, 
									 epochs = 10, 
									 validation_split = 0.15,
									 callbacks = [callback],
									 verbose = 1)

model.evaluate(x_test, y_test, batch_size=64)
#%%
'''gradient tape'''
batch_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(len(x_train)) #학습용 batch dataset은 32개

lr = 0.1
lr_scehdule = []
lr_scehdule.append(lr)
loss_function = tf.keras.losses.CategoricalCrossentropy()
for i in range(30):
    x_batch, y_batch = next(iter(batch_train))
    with tf.GradientTape() as tape:
        # 예측
        output=model(x_batch) 
        # loss 계산
        loss=loss_function(y_batch, output)
    # gradient 계산
    gradients=tape.gradient(loss, model.trainable_variables) #loss에 대해 각 trainable_variables를 편미분
    # 역전파 - weight 업데이트
    if i <= 4:
        lr_scehdule.append(lr)
        for k in range(len(model.trainable_variables)):
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr * gradients[k])    
    else:
        lr_scehdule.append(lr)
        for k in range(len(model.trainable_variables)):
            lr = lr * tf.math.exp(-0.1)
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr * gradients[k])
    print(loss)

plt.plot(lr_scehdule)
#%%
'''
LearningRateScheduler를 이용한 learning rate 조정
 - step
'''
def step_lr_scheduler(epoch):
    first_lr = 0.1
    down = 0.5
    epoch_down_cycle = 5  # 5 epoch동안 동결
    lr = first_lr * (down ** np.floor(epoch/epoch_down_cycle)) 
    print('epoch=',epoch,'lr=', lr)
    return lr

callback = tf.keras.callbacks.LearningRateScheduler(step_lr_scheduler, verbose=1)
history = model.fit(x=x_train, y=y_train, 
                    batch_size = 128, 
                    epochs = 20, 
                    validation_split = 0.15,
                    callbacks = [callback],
                    verbose = 1)
#%%
'''gradient tape'''
batch_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(len(x_train)) #학습용 batch dataset은 32개

lr = 0.1
down = 0.5
epoch_down_cycle = 5  # 5 epoch동안 동결
lr_scehdule = []
lr_scehdule.append(lr)
loss_function = tf.keras.losses.CategoricalCrossentropy()
for i in range(30):
    x_batch, y_batch = next(iter(batch_train))
    with tf.GradientTape() as tape:
        # 예측
        output=model(x_batch) 
        # loss 계산
        loss=loss_function(y_batch, output)
    # gradient 계산
    gradients=tape.gradient(loss, model.trainable_variables) #loss에 대해 각 trainable_variables를 편미분
    # 역전파 - weight 업데이트

    lr_scehdule.append(lr)  
    if i % 5 == 0:
        lr = lr * (down ** np.floor(i / epoch_down_cycle)) 
        for k in range(len(model.trainable_variables)):
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr * gradients[k])    
    print(loss)

plt.plot(lr_scehdule)
#%%
'''
 ReduceLROnPlateau를 이용한 learning rate 조정
	- 성능 향상이 없을 때 Learning rate를 동적으로 감소)
'''
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduceLR = ReduceLROnPlateau(
		monitor='val_loss',  # val_loss 기준으로 callback 호출
		factor=0.5,          # callback 호출시 학습률을 1/2로 줄임
		patience=5,          # moniter값의 개선없을 시 5번을 참아
		mode='min',          # moniter가 loss이니까 min
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
#%%
# '''
# GradientTape(그래디언트 테이프)
# 텐서플로는 자동 미분(주어진 입력 변수에 대한 연산의 그래디언트(gradient)를 계산하는 것) 을 위한 tf.GradientTape API를 제공
# tf.GradientTape는 context 안에서 실행된 모든 연산을 tape에 기록
# 그 다음 텐서플로는 후진 방식 자동 미분(reverse mode differentiation)을 사용해 tape에 "기록된" 연산의 그래디언트를 계산
# gradient 계산 후 weight update
# '''
# # compile 대신 loss function, optimizer, metircs 정의하고 학습시 모델 적용
# loss_function = tf.keras.losses.CategoricalCrossentropy()
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# train_loss = tf.keras.metrics.Mean(name='train_loss') #모든 loss의 평균
# train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


# # train_step, test_step 정의
# @tf.function # 그래프로 변환하여 성능 향상
# def train_step(x_train, y_train): # 학습에 대해 GadientTape.gradient로 미분 구해서 apply_gradient 로 weight를 update
# 	with tf.GradientTape() as tape:
# 		# 예측
# 		output=model(x_train) 
# 		# loss 계산
# 		loss=loss_function(y_train, output)
# 	# gradient 계산
# 	gradients=tape.gradient(loss, model.trainable_variables) #loss에 대해 각 trainable_variables를 편미분
# 	# 역전파 - weight 업데이트
# 	optimizer.apply_gradients(zip(gradients, model.trainable_variables))
# 	# loss, accuracy 업데이트
# 	train_loss(loss) 
# 	train_accuracy(y_train, output)


# @tf.function
# def test_step(x_test, y_test): # 성능 계산
# 	output=model(x_test)
# 	loss=loss_function(y_test, output)
	
# 	test_loss(loss)
# 	test_accuracy(y_test, output)

# # train, test dataset의 배치 사이즈 지정
# batch_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(128) #학습용 batch dataset은 32개
# batch_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


# epochs=30
# for epoch in range(epochs):
# 	train_loss.reset_states() #매 epoch마다 loss, accuracy 리셋
# 	train_accuracy.reset_states()
# 	test_loss.reset_states()
# 	test_accuracy.reset_states()
	
# 	for x_train, y_train in batch_train: #배치별로 이미지와 라벨이 들어감
# 		train_step(x_train, y_train)
# 	for x_test, y_test in batch_test: #검증
# 		test_step(x_test, y_test)
# 	print(f'epoch: {epoch +1} ,loss:{train_loss.result()}, acc: {train_accuracy.result()},\
# 		test_loss:{test_loss.result()}. test_accuracy:{test_accuracy.result()}')
	
#%%

'''
LearningRateScheduler를 이용한 learning rate 조정
 - cosine decay restarts (배치단위)
	 (초기 lr을 cosine형태로 감소하다가 다시 초기lr의 일정부분만큼 증가시키며 반복)
'''
from tensorflow.keras.experimental import CosineDecayRestarts
																																		#처음에 감소하는 스텝,
cos_decay_restarts = CosineDecayRestarts(initial_learning_rate=0.01, first_decay_steps=10, t_mul=1, m_mul=0.9, alpha=0)
steps_list = range(0, 120)
lr_list = cos_decay_restarts(steps_list)

import matplotlib.pyplot as plt
%matplotlib inline
def plot_scheduler(epoch_list, lr_list, title=None):
		plt.figure(figsize=(6,4))
		plt.plot(epoch_list, lr_list)
		plt.xlabel('epochs')
		plt.ylabel('learning rate')
		plt.title(title)

plot_scheduler(steps_list, lr_list, 'Cosine Decay Restarts')

# CosineDecay 객체는 optimizer의 learning_rate 인자로 입력되어야 함. 
model.compile(tf.keras.optimizers.Adam(learning_rate=cos_decay_restarts), loss='mse')
history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=10, validation_split=0.15) 
#%%