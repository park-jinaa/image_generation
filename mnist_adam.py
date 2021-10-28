#%%
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.gen_math_ops import log1p_eager_fallback, minimum_eager_fallback
import matplotlib.pyplot as plt
#%%
mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test)=mnist.load_data()

# # 데이터 탐색
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0],cmap='gray')
# y_train[0]
# x_train[0]

# 데이터 전처리
from tensorflow.keras.utils import to_categorical
def percentage_onehot(x,y):
    # float32로 변경, 학습성능 향상을 위해 픽셀값을 0~1 사이 값으로 변환 (0과 1사이의 연속형 변수로 바꿔주기 위함)
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

def image_model():
    input_=layers.Input(shape=(input_size_width, input_size_height))
    x = layers.Flatten()(input_)
    x = layers.Dense(100, activation='relu')(x) 
    x = layers.Dense(30, activation='relu')(x)
    output_ = layers.Dense(10, activation='softmax')(x) # 실제값 : multinomial distribution
    model=Model(inputs=input_, outputs=output_)  

    return model

model = image_model()
model.summary() 
model.weights
#%%
batch_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train), reshuffle_each_iteration=True).batch(128) #학습용 batch dataset: 128개


#%%
'''

# 1. gradient tape (모든 연산을 tape에 기록) - adam 

'''
# default setting
lr = 0.001 # initial learning rate
lr_schedule = [] # learning rate 저장
loss_function = tf.keras.losses.CategoricalCrossentropy()

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

for i in range(30):
    x_batch, y_batch = next(iter(batch_train))              # [1,128] [129,256] [257,512]
    with tf.GradientTape() as tape:
        # 예측
        output=model(x_batch)                               # x_batch를 가지고 신경망 모형 거친 후의 output
        # loss 계산
        loss=loss_function(y_batch, output)                 # output과 y의 차이에 관한 loss function을 돌림
                                    
    # gradient 계산
    gradients=tape.gradient(loss, model.trainable_variables) #loss에 대해 각 trainable_variables를 편미분 -> w,b 업데이트
    if i == 0:
        momentum = [beta1 * 0 + (1 - beta1) * g for g in gradients]
        weight = [beta2 * 0 + (1 - beta2) * tf.square(g) for g in gradients]
        modified_lr = [lr / tf.sqrt(w + epsilon) for w in weight]
        update = [lr * m for lr, m in zip(modified_lr, momentum)]
        for k in range(len(model.trainable_variables)):
            model.trainable_variables[k].assign(model.trainable_variables[k] - update[k])    
    else:
        momentum = [beta1 * m + (1 - beta1) * g for m, g in zip(momentum, gradients)]
        weight = [beta2 * w + (1 - beta2) * tf.square(g) for w, g in zip(weight, gradients)]
        modified_lr = [lr / tf.sqrt(w + epsilon) for w in weight]
        update = [lr * m for lr, m in zip(modified_lr, momentum)]
        for k in range(len(model.trainable_variables)):
            model.trainable_variables[k].assign(model.trainable_variables[k] - update[k])
    
    lr_schedule.append(modified_lr)
    print(loss)
#%%
# 30번 반복하면서, 한 번마다 6개의 trainable_variables가 존재 -> 임의로 첫 값 [0,0] or [0]

for k in range(6):
    if k % 2 == 0: # weight
        plt.plot([lr_schedule[i][k].numpy()[0, 0] for i in range(30)])
        plt.show()
        plt.close() 
    else:          # bias
        plt.plot([lr_schedule[i][k].numpy()[0] for i in range(30)])
        plt.show()
        plt.close() 


#%%
gradients[0]
print(np.sum(gradients[0].numpy()))
#%%
'''
# 2. bias 와 weight의 lr을 구분해서 지정
'''
lr1 = 0.001 # initial learning rate
lr2 = 0.001 # initial learning rate
lr_schedule1 = [] # learning rate 저장
lr_schedule1.append(lr1)
lr_schedule2 = [] # learning rate 저장
lr_schedule2.append(lr2)
loss_function = tf.keras.losses.CategoricalCrossentropy()
for i in range(20):
    x_batch, y_batch = next(iter(batch_train))              # [1,128] [129,256] [257,512]
    with tf.GradientTape() as tape:
        # 예측
        output=model(x_batch)                               # x_batch를 가지고 신경망 모형 거친 후의 output
        # loss 계산
        loss=loss_function(y_batch, output)                 # output과 y의 차이에 관한 loss function을 돌림
                                    
    # gradient 계산
    gradients=tape.gradient(loss, model.trainable_variables) #loss에 대해 각 trainable_variables를 편미분 -> w,b 업데이트
    for k in range(len(model.trainable_variables)):
        if k % 2 == 0: # weight
            lr1 = lr1 * tf.math.exp(-0.1)
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr1 * gradients[k]) 
            #print('k가 짝수. k = %d, i = %d, lr1 = %s' % (k, i, lr1)) # dummy code
            lr_schedule1.append(lr1)
        else: # bias
            lr2 = lr2 * tf.math.exp(-0.01)    
            model.trainable_variables[k].assign(model.trainable_variables[k] - lr2 * gradients[k])
            #print('k가 홀수. k = %d, i = %d, lr2 = %s' % (k, i, lr2))
            lr_schedule2.append(lr2)
    
    print(loss)

plt.plot(lr_schedule1)
plt.plot(lr_schedule2)

#%%

# a = [i for i in range(10)] ;a  # 0부터 9까지 숫자를 생성하여 i에 차곡차곡 넣겠고, i를 a라는 리스트에 넣겠다
# b = [i*5 for i in range(10)];b
# c = [i for i in range(10) if i%2==0];c #0~9숫자 중에 짝수를 리스트로 생성
# d = [i+5 for i in range(10) if i%2==1];d # 0~9숫자 중에 홀수에 5를 더해서 리스트 생성
# e = [i * j for j in range(2,10) for i in range(1,10)]; e #2단부터 9단까지 구구단을 리스트 생성
# e = [i * j for j in range(2,10)
#            for i in range(1,10)];e #j가 1일때 i가 2~10까지 반복 쭉쭉    -> 뒤부터 처리함
