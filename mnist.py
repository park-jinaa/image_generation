#%%
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.gen_math_ops import log1p_eager_fallback, minimum_eager_fallback
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
a=100
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
'''gradient tape'''
'''cycle'''
lr = 0.1
exp(-0.1) #0.9

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




'''
최초 lr
최소 lr
감소 적용할 step


최초 lr
최소 lr
첫 decay_step : 최초 decay step 수
t_mul: 전체 step수를 감안해서 얼마나 cosine annealing을 반복할지 결정하는 계수.( 1이면 전체 steps/ first_decay_steps,
                                                                     2이면 분모를 이전값 대비 2배로 증가시킴)
m_mul은 warm restart시 수행시마다 적용될 초기 learning rate
alpha는 최소 learning rate

ㅊ



def lr_scheduler(epoch, lr):
	if epoch < 5:
		return lr
	else:
		return lr * tf.math.exp(-0.1)


if lr1            =0.004 이면 최소lr까지만 하고 break하기.
   lr2=lr1
   if lr2>0.001:
        lr2* exp
'''




































#%%

 #%%


# decay_steps=30  #30스텝동안 감소
# alpha=0.0       #최소 lr
# first_lr=1e-2
# def decay_lr(step): #step 수만큼 반복
#     lr=first_lr
#     step= min(step, decay_steps) # step>decay_step이면, decay_step을 계속 적용(30번)
#     cosine_decay= 0.5 + (1+np.cos(np.pi * step / decay_steps))
#     decayed=(1-alpha)*cosine_decay+alpha
#     return first_lr*decayed
# steps_list = range(0, 30)
# lr_list = [decay_lr(step) for step in steps_list]


#%%

# initial_learning_rate는 최초 LR
#  first_decay_steps는 최초 decay step 수   #처음에 감소하는 step
# t_mul는 전체 steps수를 감안해서 얼마나 cosine annealing을 반복할 지 결정하는 계수. 1이면 전체 steps/first_decay_steps, 2이면 분모를 이전 값 대비 2배로 증가 시킴
# m_mul은 warm restart시 수행시마다 적용될 초기 learning rate
# alpha는 최소 learning rate로 설정 계수로 최소 lr은 initial learning rate * alpha로 설정됨.

from tensorflow.keras.experimental import CosineDecayRestarts

cos_decay_restarts = CosineDecayRestarts(initial_learning_rate=0.01, first_decay_steps=10, t_mul=1, m_mul=0.9, alpha=0)
                                                                      # 초기 fisrt_decay스텝과 같은 형태로 10번만에 맨밑으로 /2이면 분모를 이전 값 대비 2배로 증가 시킴 
                                                                      #           초기 initial_lr *0.9

steps_list = range(0, 120)   #batch를 120번 적용
lr_list = cos_decay_restarts(steps_list)

plot_scheduler(steps_list, lr_list, 'Cosine Decay Restarts')
#first decay step은 10
































#%%
decay_steps=30 #30스텝동안 감소
alpha=0.0      #최소 lr
first_lr=1e-2

def decay_lr(step): #step 수만큼 반복
    step= min(step, decay_steps) # step>decay_step이면, decay_step을 계속 적용(30번)
    cosine_decay= 0.5 + (1+np.cos(np.pi * step / decay_steps))
    decayed=(1-alpha)*cosine_decay+alpha
    return first_lr*decayed
steps_list = range(0, 30)
lr_list = [decay_lr(step) for step in steps_list]

plot_scheduler(steps_list, lr_list, 'Cosine Decay')
#%%

decay_steps = 10
alpha = 1e-5      #최소 lr
max_lr = first_lr = 0.1
lr_schedule = []
# lr_scehdule.append(first_lr)

loss_function = tf.keras.losses.CategoricalCrossentropy()
for i in range(50):
    x_batch, y_batch = next(iter(batch_train))
    with tf.GradientTape() as tape:
        # 예측
        output=model(x_batch) 
        # loss 계산
        loss=loss_function(y_batch, output)
    # gradient 계산
    gradients=tape.gradient(loss, model.trainable_variables) #loss에 대해 각 trainable_variables를 편미분
    # 역전파 - weight 업데이트
    if i%decay_steps == 0:
        max_lr=max_lr*0.9
        # lr_schedule.append(max_lr) 실제로 update하지 않았지만 스케쥴에 추가
        lr = max_lr # 주기와 일치하는 반복수에 도착한 경우, 현재 lr을 max_lr로 수정해줌
        for k in range(len(model.trainable_variables)):
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr * gradients[k])    
        lr_schedule.append(lr)
    else:
        # lr_schedule.append(lr)
        for k in range(len(model.trainable_variables)):
            lr= alpha + 0.5*(max_lr - alpha)*( 1 + np.cos(np.pi * (i%decay_steps) / decay_steps))  #decay_steps= 각 주기의 step 수
                                                # (i%decay_steps) = 0일 때 lr=max_lr이 되고
                                                # (i%decay_steps) = decay_steps일 때 lr=alpha가 된다.
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr * gradients[k])  
        lr_schedule.append(lr)
    print(loss)  

plt.plot(lr_schedule)

#%%
from tensorflow.keras.experimental import CosineDecayRestarts

cos_decay_restarts = CosineDecayRestarts(initial_learning_rate=0.01, first_decay_steps=10, t_mul=1, m_mul=0.9, alpha=0)
steps_list = range(0, 120)
lr_list = cos_decay_restarts(steps_list)

plot_scheduler(steps_list, lr_list, 'Cosine Decay Restarts')#%%
#%%

#Chris Deotte.
LR_START = 1e-5 #초기에 lr start
LR_MAX = 1e-2   
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 10
LR_STEP_DECAY = 0.75
lr_schedule=[]

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
    if i < LR_RAMPUP_EPOCHS:  # 5번동안 올라간다.
        lr=LR_START
        lr_schedule.append(lr)
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * i + LR_START  # 처음엔 epochs=0 -> LR_START
                                                                     #      epochs=1 -> (LR_MAX-LR_START)/5*1 + LR_START
                                                                     # ''' 4까지 올라가
        for k in range(len(model.trainable_variables)):
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr * gradients[k])    
        
    elif i < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:                  # 5~ 15 step까지 유지
        lr = LR_MAX
        lr_schedule.append(lr)
        for k in range(len(model.trainable_variables)):
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr * gradients[k])    

    else:
        lr_schedule.append(lr)
        lr = LR_MAX * LR_STEP_DECAY**((i - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//2) #
                                   #15-15 //2 =0 ->0.75의 0승
                                   #16-15 //2 =0
                                   #17-15 //2 =1 -> 0.75의 1승
                                   # 18일땐 1 , 19일땐 2, 이젠 제곱해주고 '''
        for k in range(len(model.trainable_variables)):
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr * gradients[k])    
    print(loss)  
plt.plot(lr_schedule)


