#%%
# y 지정1

n=100
p=3
k=5

from types import prepare_class
import numpy as np
import random
y=[]
for i in range(n):
    y.append(random.randrange(k))
#%%
from tensorflow.keras.utils import to_categorical
def onehot_(y):
    y=to_categorical(y)
    return y
y_indicator=onehot_(y)
#%%


# x 지정
x = np.random.normal(size=(n, p))

# alpha 지정
alpha = np.random.normal(size=(1, k-1)) #나중에 np.zero 

# beta 지정
beta = np.random.normal(size=(p, 1))

#%%
'''자동 broadcasting'''
right = np.dot(x, beta) # 100 * 1
left = alpha # 1 * 4
mat = right + left # 100 * 4
right.shape
left.shape
mat.shape

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a = np.hstack([np.ones([n, 1]), sigmoid(mat)])
b = np.hstack([sigmoid(mat), np.zeros([n, 1])])
np.log(a - b)


#%%
'''수동 broadcasting'''
left = np.dot(x,beta)
left = left1 = np.transpose(left)
for i in range(k-2):
    left = np.append(left, left1, axis=0)
left.shape

right = right1 = alpha
for i in range(n-1):
    right=np.append(right,right1,axis=1)
right.shape

mat=1 / (1 + np.exp(-left-right))
np.shape(mat)

a = np.vstack([np.zeros([1,n]),mat])
b = np.vstack([mat,np.ones([1,n])])

log_lack=np.dot(y_indicator, a-b)

'''
import math
math.log(np.dot(y_indicator, a-b))
'''
import math
math.log10(np.dot(y_indicator, a-b))


type(log_lack)
import pandas as pd
log_lack = pd.DataFrame(log_lack)

log_lack['result_column']=log_lack.iloc[:,:].sum(axis=1)
log_lack_sum=log_lack['result_column']
log_lack_sum/(-n)
# %%



'''
3. real data
'''
import pandas as pd
import numpy as np
cr = pd.read_csv('bostonhousing_ord.csv', encoding='cp949', index_col=0)
cr.head()
n,p = cr.shape
cr = cr.reset_index()
x = np.array(cr.iloc[:,1:])

#y를 0,1,2,3,4
y = np.array(cr.iloc[:,0])
y = y - 1
# temp = list(np.ones(len(y)))
# y = [a - b for a, b in zip(y,temp)]


# k = len(set(y))
k = len(np.unique(y))

from tensorflow.keras.utils import to_categorical
def _onehot(y):
    y=to_categorical(y)
    return y
y_indicator=onehot_(y)
y_indicator.shape

# aplha와 beta 지정
'''제약조건 alpha'''
alpha = np.random.normal(size=(1, k-1))
beta = np.random.normal(size=(p, 1))

right = np.dot(x, beta) # 100 * 1
mat = alpha + right

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a = np.hstack([np.ones([n, 1]), sigmoid(mat)])
b = np.hstack([sigmoid(mat), np.zeros([n, 1])])
result = np.log(a - b)
result.shape

np.sum(y_indicator * result)
#%%
left = alpha # 1 * 4
mat = right + left # 100 * 4
right.shape
left.shape
mat.shape

#%%
   def __init__(self, X, y):
        super(BuildModel, self).__init__() # 상속이 이루어지는 부분
        
        X = (X - tf.math.reduce_mean(X, axis=0)) / tf.math.reduce_std(X, axis=0) # scaling
        self.K = len(tf.unique(y)[0])                 # number of category
        self.y_true = y
        y = tf.keras.utils.to_categorical(y)  # one-hot encoding
        self.y = tf.cast(y, tf.float32)
        self.X = tf.cast(X, tf.float32)
        self.n, self.p = X.shape

    def loss(self, mat1, mat2):
        loss = -tf.reduce_sum(tf.multiply(self.y, tf.math.log(mat1 - mat2 + 1e-8))) / self.n
        return loss

        
  def repa(self, para):                                         #para는 alpha네
    '''reparametrization for positive condition'''
    theta = []
    theta.append(para[0, 0])
    for i in range(1, self.K-1):
        theta.append(tf.square(para[0, i]) + theta[i-1])
    theta = tf.stack(theta)[tf.newaxis, :]
    return theta
#%%

alpha[0,0]

#re1.sum(axis=1).sum(axis=0)



para = alpha
K = 5
theta = []
theta.append(para[0, 0])
for i in range(1, K-1):
    theta.append(tf.square(para[0, i]) + theta[i-1])
theta = tf.stack(theta)[tf.newaxis, :]






'''
'''

#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%
class BuildModel(tf.keras.models.Model): # 부모 class
    def __init__(self, X, y):
        super(BuildModel, self).__init__() # 상속이 이루어지는 부분
        
        X = (X - tf.math.reduce_mean(X, axis=0)) / tf.math.reduce_std(X, axis=0) # scaling
        self.K = len(tf.unique(y)[0])                 # number of category
        self.y_true = y
        y = tf.keras.utils.to_categorical(y)  # one-hot encoding
        self.y = tf.cast(y, tf.float32)
        self.X = tf.cast(X, tf.float32)
        self.n, self.p = X.shape

    def loss(self, mat1, mat2):
        loss = -tf.reduce_sum(tf.multiply(self.y, tf.math.log(mat1 - mat2 + 1e-8))) / self.n
        return loss

    def repa(self, para):
        '''reparametrization for positive condition'''
        theta = []
        theta.append(para[0, 0])
        for i in range(1, self.K-1):
            theta.append(tf.square(para[0, i]) + theta[i-1])
        theta = tf.stack(theta)[tf.newaxis, :]

        return theta

    def init_para(self): 
        alpha = tf.Variable(tf.random.normal([1,self.K-1], 0, 1))
        beta = tf.Variable(tf.random.normal([self.p, 1], 0, 1))
        return alpha, beta

    def learn(self, iteration = 300, lr = 0.3):
        
        alpha , beta = self.init_para()
        lr = lr
        for j in tqdm(range(iteration)):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(alpha)
                tape.watch(beta)
                
                mat1, mat2, theta = self.compute(alpha, beta)
                loss = self.loss(mat1, mat2)
            
            # update
            grad1 = tape.gradient(loss, alpha)
            grad2 = tape.gradient(loss, beta)
            alpha = alpha - lr * grad1
            beta = beta - lr * grad2
        # theta = np.array(theta)
        theta = theta.numpy()
        beta = beta.numpy()
        return mat1, mat2, theta, beta

    def learn2(self, grad_norm = 1e-6, lr = 0.3):
        
        alpha, beta = self.init_para()
        lr, grad, iteration = lr, 1, 0
        while grad > grad_norm:
            iteration += 1
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(alpha)
                tape.watch(beta)
                
                mat1, mat2, theta = self.compute(alpha, beta)
                loss = self.loss(mat1, mat2)
            
            # update
            grad1 = tape.gradient(loss, alpha)
            grad2 = tape.gradient(loss, beta)
            grad = tf.reduce_sum(tf.concat([grad1**2, tf.transpose(grad2)**2], axis=1))

            alpha = alpha - lr * grad1
            beta = beta - lr * grad2
        theta = theta.numpy()
        beta = beta.numpy()
        print("iteration : " + str(iteration))
        
        return mat1, mat2, theta, beta
    
    def true_pred(self, mat1, mat2):
        y_true = self.y_true
        y_pred = tf.argmax(mat1 - mat2, axis=1)
        y_pred = y_pred.numpy()
        return y_true, y_pred 

    def con_table(self, y_true, y_pred):
        table = pd.crosstab(y_true, y_pred,rownames=['True'], colnames=['Predicted'], margins=True)
        return table
        
    def acc(self, y_true, y_pred):
        table = self.con_table(y_true, y_pred)
        acc = np.sum(np.diag(table)[:-1])/self.n
        return acc

    def MSE(self, y_true, y_pred):
        mse = np.mean(np.square(y_true - y_pred)) 
        return mse

    def compute(self, alpha, beta):

        theta = self.repa(alpha)
        mat1 = tf.nn.sigmoid(theta + tf.matmul(self.X, beta))
        mat1 = tf.concat((mat1, tf.ones((self.n, 1))), axis=-1)
        mat2 = tf.nn.sigmoid(theta + tf.matmul(self.X, beta))
        mat2 = tf.concat((tf.zeros((self.n, 1)), mat2), axis=-1)

        return mat1, mat2, theta

#%%
data = pd.read_csv(r'C:\Users\uos\iCloudDrive\ordinal\bostonhousing_ord.csv')
# data = pd.read_csv(r'C:\Users\uos\iCloudDrive\ordinal\stock_ord.csv')
X = np.array(data.iloc[:, 1:])        # design matrix
y = np.array(data.iloc[:,0]) -1

model = BuildModel(X = X, y = y)

# 출력
mat1, mat2, theta, beta = model.learn2(grad_norm = 1e-4, lr = 0.3)
mat1, mat2, theta, beta = model.learn(iteration = 500, lr = 0.3)
y_true, y_pred = model.true_pred(mat1, mat2)

loss = model.loss(mat1, mat2)
confusion_table = model.con_table(y_true, y_pred)
accurucy = model.acc(y_true, y_pred)
MSE = model.MSE(y_true, y_pred)

# ad = np.array(confusion_table)
# np.vectorize(np.array(confusion_table))
# np.array(confusion_table)


#%%

plt.figure(figsize=(20,10))
plt.subplot(121)
plt.hist(y_pred,bins=model.K, range=[-0.5, 4.5], alpha=0.33,label='pred',color='blue')
plt.hist(y_true,bins=model.K, range=[-0.5, 4.5], alpha=0.33,label='true',color='orange')
plt.title('Boston Housing - Validation - Output Distribution',fontsize=14)
plt.xlabel('PREDICTED',fontsize=16)
plt.ylabel('COUNT',fontsize=16)
plt.legend()
plt.tight_layout()

# %%
