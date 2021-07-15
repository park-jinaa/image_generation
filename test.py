#%%
def fine_f(w_x):
    result=w_x+2
    return result

def fine_f(x1,x2):
    return x1-x2
fine_f(x2=3,x1=1)

def fine_f(x,y,const=1):
    return (x+y)*const

def f(x,y):
    return x+y
c=f(3,4)
#%%
"""
Copyright 2016 Marc Garcia <garcia.marc@gmail.com>
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import functools
import operator
import gzip
import struct
import array
import tempfile
from types import prepare_class
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve  # py2
try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin
import numpy


__version__ = '0.2.2'


# `datasets_url` and `temporary_dir` can be set by the user using:
# >>> mnist.datasets_url = 'http://my.mnist.url'
# >>> mnist.temporary_dir = lambda: '/tmp/mnist'
datasets_url = 'http://yann.lecun.com/exdb/mnist/'
temporary_dir = tempfile.gettempdir


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


def download_file(fname, target_dir=None, force=False):
    """Download fname from the datasets_url, and save it to target_dir,
    unless the file already exists, and force is False.
    Parameters
    ----------
    fname : str
        Name of the file to download
    target_dir : str
        Directory where to store the file
    force : bool
        Force downloading the file, if it already exists
    Returns
    -------
    fname : str
        Full path of the downloaded file
    """
    target_dir = target_dir or temporary_dir()
    target_fname = os.path.join(target_dir, fname)

    if force or not os.path.isfile(target_fname):
        url = urljoin(datasets_url, fname)
        urlretrieve(url, target_fname)

    return target_fname


def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.
    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse
    endian : str
        Byte order of the IDX file. See [1] for available options
    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https://docs.python.org/3/library/struct.html
        #byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, '
                             'file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, '
                             'file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type '
                             '0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items,
                                                          len(data)))

    return numpy.array(data).reshape(dimension_sizes)


def download_and_parse_mnist_file(fname, target_dir=None, force=False):
    """Download the IDX file named fname from the URL specified in dataset_url
    and return it as a numpy array.
    Parameters
    ----------
    fname : str
        File name to download and parse
    target_dir : str
        Directory where to store the file
    force : bool
        Force downloading the file, if it already exists
    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    """
    fname = download_file(fname, target_dir=target_dir, force=force)
    fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
    with fopen(fname, 'rb') as fd:
        return parse_idx(fd)


def train_images():
    """Return train images from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.
    Returns
    -------
    train_images : numpy.ndarray
        Numpy array with the images in the train MNIST database. The first
        dimension indexes each sample, while the other two index rows and
        columns of the image
    """
    return download_and_parse_mnist_file('train-images-idx3-ubyte.gz')


def test_images():
    """Return test images from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.
    Returns
    -------
    test_images : numpy.ndarray
        Numpy array with the images in the train MNIST database. The first
        dimension indexes each sample, while the other two index rows and
        columns of the image
    """
    return download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')


def train_labels():
    """Return train labels from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.
    Returns
    -------
    train_labels : numpy.ndarray
        Numpy array with the labels 0 to 9 in the train MNIST database.
    """
    return download_and_parse_mnist_file('train-labels-idx1-ubyte.gz')


def test_labels():
    """Return test labels from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.
    Returns
    -------
    test_labels : numpy.ndarray
        Numpy array with the labels 0 to 9 in the train MNIST database.
    """
    return download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')



# c규약, 틀(설계도) 

#%%
a=[1,2,3,4,5]
b=[111]
30 in a
30 not in a
'p' in 'hello python'
a+b
b*3

#%%
'''
 # class
구조: 명사(attribute, property, instance 변수), 동사(method)
class 클이름:
    명사(attribute)를 초기화 하는 공간
    동사를 정의하는 공간

'''

##클래스 정의/호출(= 객체 만들기)
#behavior만 생각했을 때
class SoccerPlayer:
    def shoot(self):  #self 필수
        print('슛을 날리다')
        
    def pass_the_ball(self,x,y):
        print('패스를 합니다')
        
player=SoccerPlayer() #객체 생성
    #호출하면 설계도로 실체있는 집으로
    #축구선수라는 추상화된 개념을 실존캐릭으로 생성
player  #어떤 클래스로 만들어졌는지, 램의 주소값 뭔지
player.shoot()
player.pass_the_ball(1,2)


player1=SoccerPlayer() # 기능은 비슷하지만 외모 특성은 같음
player2=SoccerPlayer()

player1.shoot()
player2.shoot()

def let_player1_sjhoot():
    print('palyer1이 슛을 하게 합니다')
def let_player2_sjhoot():
    print('palyer2가 슛을 하게 합니다')
let_player1_sjhoot() #sequential programing 직관적이지 못함


#attribute 초기화(feat 생성자)
class SoccerPlayer:
    def __init__(self):#객체 생성시점에서 호출되는 생성자. 이렇게 정의하면 init을 덮어씌움
                       #따로 명시안하면 init이 정의되긴 하지만 놀랍게도 아무일도 일어나지 않았다
        print('난 태어남')
    def shoot(self):
        print('슛을 날리다')
player1= SoccerPlayer()

class SoccerPlayer:
    def __init__(self, height, weight):
        print('난 태어남')
        self.fine_height =height  #생성될때 해야하는 일들을 호출해보자
        self.fine_weight = weight
    def shoot(self):
        print('슛을 날리다')
player1= SoccerPlayer(180, 50) # 집 만들 때 설계도(뼈대)는 같지만 지붕색깔, 창문크기는 바뀜.
player2= SoccerPlayer(190, 70) # 그에 해당하는 인스턴스가 바뀌는 것임
player1.fine_height   #attribute도 method처럼 '객체.attribute'와 같이 .으로 가져올 수 있음
player2.fine_height


#self 존재이유
class SoccerPlayer:
    def __init__(self, height, weight):
        print('난 태어남')
        self.fine_height =height 
        self.fine_weight = weight
    def shoot(self):
        self.fine_height =self.fine_height+1
        print('슛을 날리다')
player1=SoccerPlayer(180,70)
player1.fine_height
player1.shoot()
player1.fine_height

player2=SoccerPlayer(180,70)
player2.fine_height
player2.shoot()
player2.fine_height
player1.fine_height


#상속
class Human:
    def __init__(self, weight, height):
        self.fine_weight=weight
        self.fine_height=height
    def walk(self):
        print('걷는다')
h1=Human(60,170)
h1.walk()

class Athlete:
    def __init__(self, weight, height):
        self.fine_weight=weight
        self.fine_height=height
    def walk(self):
        print('걷는다')
    def workout(self):
        print('운동을 합니다')
        
class Athlete(Human):  #상속. 위와 동치
    def workout(self):
        print('운동을 합니다')
    
h2=Athlete(50,100)
h2.walk()

class Athlete(Human):  
    def __init__(self, weight, height, fat_rate):  #부모 생성자에 fat_rate 추가할거다
        super().__init__(weight,height)  # 2개는 부모의 해당 함수를 호출하겠다
        self.fat_rate=fat_rate
                
    def workout(self):
        print('운동을 합니다')
h3=Athlete(50,180,11)
h3.walk()

class SoccerPlayer(Athlete):  # 상속을 여러 번

    def workout(self):
        print('축구를 한다')   #부모의 workout을 overwriting(같은 행위지만 다르게 행동)
h4=SoccerPlayer(50,100,11)      
h4.walk()
h4.workout()


# 클래스 관점에서의 파이썬 기본 자료형
c= {'banana':1200,'blue':100}
c.keys()


'''
파이썬에서 다루는 모든 자료형(변수)은 클래스의 객체(클래스의 인스턴스화된 변수)
객체: 소프트웨어 세계에 구현할 대상
    (=현실의 대상과 비슷하여 상태나 행동 등을 가지지만 소웨관점에선 그저 콘셉트->그이상의 사고필요)
클래스: 이를 구현하기 위한 설계도
인스턴스: 이 설계도에 따라(클래스 정의를 통해) 소프트웨어 세계에 구현된(만들어진) 실체
인스턴스화: 설계도를 바탕으로 객체를 소프트웨어에 실체화
api: 프로그래밍 언어, 라이브러리, 어플리케이션, 메소드 등이 제공하는 기능들을 
    제어할 수 있게 만든 인터페이스 혹은 관련 문서 혹은 page (환경 모르고 제어 가능)

a = ClassName() 객체화 해줌
b=[1,2,3] 기본 자료형같은 경우 기호로 객체화해줌
b=list([1,2,3]) 괄호 이용해서 list생성 = 클래스 이름으로 호출해서 객체를 만든 것
b. 하면 메소드들이 나온다.
   메소드들(인터페이스들) = 위에서 클래스 정의했을 때의 def함수들이라 생각 
    list라는 클래스에 그러한 api가 있다.
c= {'banana':1200,'blue':100}
c.keys()

dic 클래스에는 keys라는 api가 있고, list라는 클래스에는 append라는 함수가 있다. x
각각의 자료형들은 클래스의 객체인데, list클래스와 dic클래스 등이 있는데,
list와 dic은 각각의 자료형 특징에 맞는 규약으로 만들어진 클래스이고, 클래스 안에는
얘네들이 데이터를 저장하는 방식에 대해 쉽게 다룰수 있고 
얘네들을 조작할수있는 인터페이스를 메소드나 어트리뷰트 또는 프로퍼티 형식으로 제공한다
확인하려면 인스턴스 만들어서 . 찍으면 확인할 수 있다.

필요한 기능 있을 때마다 구글링 통해 클래스의 api문서 읽어보고
    어떤 기능을 제공하는지 캐치해서 활용하는 연습

b. 이 함수들이 왜 변수에서 쓸 수 있는지 
클래스는 이런 인터페이스를 제공하도록 규약화. 이런 규약들이 존재하고 직접 정의할수도, 제공해줄수도 있다.

결국 단순 데이터 저장하는 공간이 아니라 클래스이기 때문에 
어떤 인터페이스나 메소드 어트리뷰트 사용하면 코딩을 좀더 쉽게할수있겠구나

'''


# 객체를 인스턴스 변수로 가지고 있기
# class SoccerPlayer:
#     def __init__(self, historical_weight_list):
#         self.hist_weight=historical_weight_list
# a=SoccerPlayer([100,90,80,85]) #string,dictionary,클래스의 인스턴스 전달할 수 도 있음
# a.hist_weight
# a=[100,90,80,85]

class SoccerCoach:
    def __init__(self, num_career_year):
        self.n_year=num_career_year
class Team:
    def __init__(self, coach, player_list):
        self.fine_coach=coach
        self.p_list=player_list
        
class SoccerPlayer:
    def __init__(self, weight):
        self.weight=weight
player1=SoccerPlayer(70)
player2=SoccerPlayer(80)
coach1=SoccerCoach(10)
team1=Team(coach=coach1, player_list=[player1,player2])
team1.fine_coach.n_year

#보기 쉽게
team1= Team(
    coach=coach1,
    player_list=[
        player1,
        player2
    ]
)
#python -> pep8, pep8 auto format 해주는 라이브러리: https://github.com/hhatto/autopep8


#객체 메소드의 cascading . . .
class SoccerPlayer:
    def __init__(self,weight):
        self.weight=weight
    def walk(self):
        print('걷는다')

team2=Team(
    coach=coach1,
    player_list=[
        SoccerPlayer(70),
        SoccerPlayer(80)
    ]
)
team2.p_list[0].walk()  #0번째 player의 walk
''' 팀2의 player_list를 가져와서 list의 첫원소를 꺼내고, 첫 원소의 walk를 호출함
type(team2): Team(클래스)
type(team2.player_list):리스트
type(team2.player_list[0]):SoccerPlayer(클래스)
'''

'박카스' in'박카스 한잔'

for i in range(1,100,2):
    print('룰루')
# range()를 건드리지 않고 짝수 출력하기
for tmp_var in range(50, 70):
    if tmp_var%2==0:
        print(tmp_var)



# 반복문 break/ continue  종이와 펜 연습
#break
for i in range(10):
    if i==5:
        break
    else:
        print(i)

for i in range(10):
    print(i)
    if i==5:
        break

for i in range(10):
    print(i)
    if i==5:
        break
print('ㅎ')

#continue
for i in range(10):
    if i==5:
        continue  #if가 참이면 continue밑을 패스하고 continue
    print(i)

for i in range(10):
    if i<5:
        continue
    print(i)

for i in range(10):
    print(i)
    if i<5:
        continue
    elif i==7:
        break
#list comprehension 코드 간결화
my_list=[]
for i in range(1,11):
    my_list.append(i)

[tmp for tmp in range(1,11)]

my_list = []
for i in range(1, 11):
    if i % 2 == 0:
        my_list.append(i)

[tmp for tmp in range(1,11) if tmp%2==0]



# while
# true일 때까지 돌음
# while(비교연산): #비교연산 혹은 boolen condition의 변수를 변경하는 로직이 포함되어야 함
#     구문1
#     구문2
a=0
while a<5:
    print(a)
    a=a+1
    
a = 0
while a <= 5:
    a = a + 1
    print(a)
print("ㅋ")

a = 1
while a <= 5 or a > 2:
    a = a + 1
    if a == 3:
        continue
    
    if a == 10:
        break
    print(a)
    
#for예제를 while로 변경하기
for i in range(1, 11, 2):
    print(i)

i=1
while i<11:
    if i%2==1:
        print(i)
    i+=1

# while True:
#     current_time = 현재시간을 받아온다
#     if current_time >= 09:00am and current_time < 09:01am:
#         날씨데이터를 웹에서 긁어서 나한테 이메일을 보내게 한다
#         break
#     else:
#         time.sleep(10)


#모듈 : 다양한 클래스와 함수들을 모아놓은 것, 라이브러리라고도 함






##%%



#%%
#%%
'''
#ndarray: N차원(dimension) 배열(array)
import numpy as np
array1=np.array([1,2,3])    
array2=np.array([[1,2,3],[2,3,4]])
array1.shape  #요소 3개
array2.shape
array1.ndim
array2.ndim
#ndarray의 datatype은 같은 데이터타입만 가능. 숫자, 문자, 불 타입 다 가능
array1.dtype

astype()
변경을 원하는 타입을 입력.
대용량 데이터를 ndarray로 만들 때 메모리를 절약하기 위해 사용
0,1,2정도 숫자는 64비트 float형 보다는 8비트, 16비트로
astype(array1)
#axis0은 행방향, axis1은 열방향
'''
import numpy as np
array1 = np.array([1,2,3])
print('array1 type:',type(array1))
print('array1 array 형태:',array1.shape)

array2 = np.array([[1,2,3],
                  [2,3,4]])
print('array2 type:',type(array2))
print('array2 array 형태:',array2.shape)

array3 = np.array([[1,2,3]]) #내포된 리스트
print('array3 type:',type(array3))
print('array3 array 형태:',array3.shape)
print('array1: {:0}차원, array2: {:1}차원, array3: {:2}차원'.format(array1.ndim,array2.ndim,array3.ndim))
    # 행축이 명확히 1이므로 2차원이 맞음


list1 = [1,2,3]
print(type(list1))
array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)


list2 = [1, 2, 'test']  #리스트는 서로다른 데이터 타입으로 가능
array2 = np.array(list2)
print(array2, array2.dtype) #정수형 값들이 문자형으로 바뀜

list3 = [1, 2, 3.0]
array3 = np.array(list3)
print(array3, array3.dtype)

# astype을 이용한 타입 변환
array_int = np.array([1, 2, 3])
array_float = array_int.astype('float64')
print(array_float, array_float.dtype)

array_int1= array_float.astype('int32')
print(array_int1, array_int1.dtype)

array_float1 = np.array([1.1, 2.1, 3.1])
array_int2= array_float1.astype('int32')
print(array_int2, array_int2.dtype)

#ndarray에서 axis 기반의 연산함수 수행
array2=np.array([[1,2,3],
                 [4,5,6]])
print(array2.sum())
print(array2.sum(axis=0))
print(array2.sum(axis=1))


###
###

# ndarray를 편리하게 생성하기 - arange, zeros, ones
sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)


zero_array = np.zeros((3,2),dtype='int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)

one_array = np.ones((3,2))
print(one_array)
print(one_array.dtype, one_array.shape)  # dtype명시 안해서 float임

# reshape
array1 = np.arange(10)
print('array1:\n', array1)

array2 = array1.reshape(2,5)
print('array2:\n',array2)

array3 = array1.reshape(5,2)
print('array3:\n',array3)

array3 = array1.reshape(-1,2)
print('array3:\n',array3)

# 변환될 수 없는 shape구조 입력하면 오류 발생
array1.reshape(4,3)

#reshape에 -1 인자값을 부여하여 특정 차원으로 고정된 가변적인 ndarray형태 변환
array1 = np.arange(10)
print(array1)
    #열 고정
array2 = array1.reshape(-1,5)  #5는 고정 그에 맞게 행이 지정됨
print('array2 shape:',array2.shape)
print('array2:\n',array2)
    #1차원으로 고정  .reshape(-1,)
array3 = array1.reshape(5,-1)
print('array3 shape:',array3.shape)
print('array3:\n',array3)


# reshape는 (-1,1), (-1,)와 같은 형태로 주로 사용됨
# 1차원  ndarray를 2차원으로 또는 2차원 ndarray를 1차원으로 변환 시 사용.
array1= np.arange(5)

#1차원 ndarray를 2차원으로 변환하되, 컬럼axis 크기는 반드시 1이어야 함
array2d_1=array1.reshape(-1,1)
print('array2d_1.shape: ',array2d_1.shape)
print('array2d_1:\n',array2d_1)

#2차원 ndarray를 1차원으로 변환
array2d=array1.reshape(-1,)
print('array2d.shape: ',array2d.shape)
print('array2d:\n',array2d)

# -1을 적용하여도 변환이 불가능한 형태로의 변환을 요구할 경우 오류 발생
array1 = np.arange(10)
array4 = array1.reshape(-1,4)


#반드시 -1값은 한개의 인자만 입력해야 함
array1.reshape(-1,-1)


array1 = np.arange(8)
array3d = array1.reshape((2,2,2))
print('array3d:\n',array3d.tolist())

# 3차원 ndarray를 2차원 ndarray로 변환
array5 = array3d.reshape(-1,1)
print('array5:\n',array5.tolist())
print('array5 shape:',array5.shape)

# 1차원 ndarray를 2차원 ndarray로 변환
array6 = array1.reshape(-1,1)
print('array6:\n',array6.tolist())
print('array6 shape:',array6.shape)
    
    
'''
ndarray의 데이터 세트 선택하기 - 인덱싱
1. 특정 위치의 단일값 추출
    array1[3]  array1[-2]  array2d[0,0]  array2d[2,1]
2. 슬라이싱 
    array1[:]  array1[:3] -012만 array2d[0:2, 0:2] -1245  array2d[:2,1:] -2356
3. 팬시 인덱싱 : 일정 인덱싱 집합을 리스트/ndarray형태로 지정->해당위치 ndarray반환
    array2d[[0,1],2] -36  array2d[[0,1]] -행인덱스만 존재 123456
    slicing과 유사하고, 불연속적인 값 가져올 수 있음
*4. 불린 인덱싱 : True/False값 인덱싱 집합 기반으로 True해당위치 ndarray반환
    ndarray내의 값이 5보다 큰 ndarray를 추출하고자 한다
    array1[array1>5]
'''
# indecing
# 특정 위치의단일값 추출
# 1에서 부터 9 까지의 1차원 ndarray 생성 
array1 = np.arange(1, 10)
print('array1:',array1)
# index는 0 부터 시작하므로 array1[2]는 3번째 index 위치의 데이터 값을 의미
value = array1[2]
print('value:',value)
print(type(value))

print('맨 뒤의 값:',array1[-1], ', 맨 뒤에서 두번째 값:',array1[-2])

array1[0] = 9 #업데이트
array1[8] = 0
print('array1:',array1)


array1d = np.arange(1, 10)
array2d = array1d.reshape(3,3)
print(array2d)

print('(row=0,col=0) index 가리키는 값:', array2d[0,0] )
print('(row=0,col=1) index 가리키는 값:', array2d[0,1] )
print('(row=1,col=0) index 가리키는 값:', array2d[1,0] )
print('(row=2,col=2) index 가리키는 값:', array2d[2,2] )


# slicing
array1 = np.arange(1, 10)
array3 = array1[:3]
print(array3)
print(type(array3))

array4 = array1[3:]
print(array4)

array5 = array1[:]
print(array5)

array1d = np.arange(1, 10)
array2d = array1d.reshape(3,3)
print('array2d:\n',array2d)

print('array2d[0:2, 0:2] \n', array2d[0:2, 0:2])
print('array2d[1:3, 0:3] \n', array2d[1:3, 0:3])
print('array2d[1:3, :] \n', array2d[1:3, :])
print('array2d[:, :] \n', array2d[:, :])
print('array2d[:2, 1:] \n', array2d[:2, 1:])
print('array2d[:2, 0] \n', array2d[:2, 0])


print(array2d[0])
print(array2d[1])
print('array2d[0] shape:', array2d[0].shape, 'array2d[1] shape:', array2d[1].shape )

# fancy indexing
array1d = np.arange(1, 10)
array2d = array1d.reshape(3,3)

array3 = array2d[[0,1], 2]
print('array2d[[0,1], 2] => ',array3.tolist())

array4 = array2d[[0,1], 0:2]
print('array2d[[0,1], 0:2] => ',array4.tolist())

array5 = array2d[[0,1]]
print('array2d[[0,1]] => ',array5.tolist())


#boolean indexing
array1d = np.arange(1, 10)
# [ ] 안에 array1d > 5 Boolean indexing을 적용 
array3 = array1d[array1d > 5]
print('array1d > 5 불린 인덱싱 결과 값 :', array3)

array1d > 5
type(array1d > 5)  #ndarray로 타입 바뀜


boolean_indexes = np.array([False, False, False, False, False,  True,  True,  True,  True])
array3 = array1d[boolean_indexes]
print('불린 인덱스로 필터링 결과 :', array3)


indexes = np.array([5,6,7,8])
array4 = array1d[ indexes ]
print('일반 인덱스로 필터링 결과 :',array4)

'''
# sort()
    np.sort(): 원 행렬은 그대로 유지한 채 원 행렬의 정렬된 행렬을 반환
    ndarray.sort(): 원 행렬 자체를 정렬한 현태로 변환하여 반환 값은 None
    내림차순으로 정렬하려면 [::-1]
# 2차원 배열에서 axis 기반의 sort()
    axis=0에 대해 sort하면 행들에 대해 오름차순
#argsort(): 원본 행렬 정렬 시 정렬된 행렬의 원래 인덱스를 필요로 할 때,
         np.argsort()를 이용한다. 정렬 행렬의 원본 행렬 인덱스를 ndarray 형으로 변환한다.
        A반의 학생이름과 점수가 있다-> 점수를 기반으로 학생들의 순위를 알고 싶다.
#np.dot(A,B): 내적
#np.transpose(A) : T

'''
# 행렬의 정렬 – sort( )와 argsort( )
## 행렬 정렬
org_array = np.array([ 3, 1, 9, 5]) 
print('원본 행렬:', org_array)
# np.sort( )로 정렬 
sort_array1 = np.sort(org_array)         
print ('np.sort( ) 호출 후 반환된 정렬 행렬:', sort_array1) 
print('np.sort( ) 호출 후 원본 행렬:', org_array)
# ndarray.sort( )로 정렬
sort_array2 = org_array.sort() #원본 행렬이 변하게 됨. 이런식으로 호출됐을 때 None이 반환됨
print('org_array.sort( ) 호출 후 반환된 행렬:', sort_array2) 
        #None이 반환됨(원본행렬이 변형됐다)
print('org_array.sort( ) 호출 후 원본 행렬:', org_array)


sort_array1_desc = np.sort(org_array)[::-1]
print ('내림차순으로 정렬:', sort_array1_desc) 


array2d = np.array([[8, 12], 
                   [7, 1 ]])

sort_array2d_axis0 = np.sort(array2d, axis=0)
print('로우 방향으로 정렬:\n', sort_array2d_axis0)

sort_array2d_axis1 = np.sort(array2d, axis=1)
print('컬럼 방향으로 정렬:\n', sort_array2d_axis1)

# key_value 형태의 데이터를 John=78, Mike=95, Sarah=84, Kate=98, Samuel=88을
# ndarray로 만들고 argsort()를 이용하여 key값을 정렬
name_array=np.array(['John','Mike','Sarah','Kate', 'Samuel'])
score_array=np.array([78,95,84,98,88])

# score_array의 정렬된 값에 해당하는 원본 행렬 위치 인덱스 반환하고
# 이를 이용하여 name_array에서 name값 호출
sort_indices = np.argsort(score_array)
print(type(sort_indices))
print('sort_indices:', sort_indices) #못한 점수의 원래 index 확인

name_array_sort=name_array[sort_indices]
name_array_sort=name_array[0,2,3]

score_array_sort=score_array[sort_indices]
print(name_array_sort)
print(score_array_sort)


# org_array = np.array([ 3, 1, 9, 5]) 
# sort_indices_desc = np.argsort(org_array)[::-1]
# print('행렬 내림차순 정렬 시 원본 행렬의 인덱스:', sort_indices_desc)


#선형대수 연산 -행렬 내적과 전치행렬 구하기
## 행렬 내적
A = np.array([[1, 2, 3],
              [4, 5, 6]])
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])

dot_product = np.dot(A, B)
print('행렬 내적 결과:\n', dot_product)

# %%
'''
# pandas: 특히 시계열 데이터에서 편리
    dataframe:column과 row의 2차원 데이터 셋
    index: dataframe/series의 고유한 key값 객체
    series: 1개의 column값으로만 구성된 1차원 데이터셋
기본 API- read_csv() head() shape info() describe() Value_counts() Sort_values()
'''
import pandas as pd
titanic_df=pd.read_csv('C:/Users/uos/Desktop/data/titanic.csv')
titanic_df.head()
titanic_df.shape
titanic_df.info()#컬럼명, 데이터 타입, 널 건수, 데이터 건수
titanic_df.describe()#평균, 표준편차, 4분위 분포도


###
### dataframe 생성, 컬럼추가, 인덱스 할당
###
#dataframe의 생성
dic1={'Name':['Ghulmin','Eunkyung','Jinwoong','Soobeon'],
      'Year':[2011,2016,2015,2015],
      'Gender':['Male','Female','Male','Male']}
#딕셔너리를 DataFrame으로 변환
data_df=pd.DataFrame(dic1)
data_df

#새로운 컬럼명을 추가
data_df=pd.DataFrame(dic1,columns=['Name','Year','Gender','Age'])
data_df

#인덱스를 새로운 값으로 할당
data_df=pd.DataFrame(dic1, index=['one','two','three','four'])
data_df

#DataFrame의 컬럼명과 인덱스
print(titanic_df.columns)
print(titanic_df.index)
print(titanic_df.index.values)


###dataframe에서 Series추출 및 DataFrame 필터링 추출
# DataFrame객체에서 []연산자 내에 한개의 컬럼만 입력하면 Series 객체를 반환
series=titanic_df['Name']
print(series.head(3))
print(type(series))

#DataFrame객체에서 []연산자 내에 여러개의 컬럼을 리스트로 입력하면 그 컬럼들로 구성된 DataFrame반환
filtered_df=titanic_df[['Name','Age']]
print(filtered_df.head(3))
print(type(filtered_df))

#DataFrame객체에서 []연산자 내에 한개의 컬럼을 리스트로 입력하면 한개의 컬럼으로 구성된 DataFrame반환
one_col_df=titanic_df[['Name']]
print(one_col_df.head(3))
print(type(one_col_df))





'''
DataFrame과 리스트, 딕셔너리, 넘파이 ndarray 상호 변환
- 리스트를 DataFrame으로 변환
    df_list1=pd.DataFrame(list1, columns=col_name1) 
    이렇게 DataFrame생성 인자로 리스트와 매핑되는 컬럼명들을 입력
- ndarray를 DataFrame으로 변환
    df_list2=pd.DataFrame(list2,columns=col_name2)
    와 같이 DataFrame 생성 인자로 ndarray와 매핑되는 컬럼명들을 입력
- 딕셔너리를 DataFrame으로 변환
    dict={'col1':[1,11],'col2':[2,22],'col3':[3,33]}
    df_dict=pd.DataFrame(dict)
    와 같이 딕셔너리의 키(key)로 컬럼명을, 값(value)을 리스트 형식으로 입력
- DataFrame을 ndarray로 변환
    DataFrame객체의 values속성을 이용해 ndarray변환
- DataFrame을 리스트로 변환
    DataFrame객체의 values속성을 이용해 ndarray변환 -> to_list()로 list변환
- DataFrame을 딕셔너리로 변환
    DataFrame객체의 to_dict()를 이용해 변환

DataFrame의 컬럼 데이터 셋 Access- 코드로 확인

DataFrame.drop(labels=None, axis=0, index=None, columns=None, inplace=False, errors='raise')
- axis=0이면 행삭제, axis=1이면 열삭제
- 원본은 유지하고 드롭된 DataFrame을 새롭게 객체 변수로 받으려면 inplace=False
  ex) titanic_drop_df=titanic_df.drop('Age_0',axis=1,inplace=False)
  ex) titanic_df= titanic_df.drop('Age_0',axis=1,inplace=False)

index
- 판다스의 index객체는 RDBMS의 PK(Primary Key)와 유사하게 DataFrame,Series의 레코드를 고유 식별하는 객체
- DataFrame, Series에서 index객체만 추출하려면 DataFrame.index 또는 Series.index속성을 통해 가능
- Series객체는 index객체를 포함하지만 Series객체에 연산 함수를 적용할 때 index는 연산에서 제외(오직 식별)
- DataFrame 및 Series에 reset_index() 메서드를 수행하면 새롭게 인덱스를 연속 숫자형으로 할당하며 기존 인덱스는 'index'라는 새로운 칼럼명으로 추가
'''
###
### DataFrame과 리스트, 딕셔너리, 넘파이 ndarray상호변환
###

### 리스트, ndarray를 DataFrame으로 변환
import numpy as np
col_name1=['col1']
list1=[1,2,3]
df_list1=pd.DataFrame(list1, columns=col_name1)
print(df_list1)

array1=np.array(list1)
print(array1.shape)
df_array1=pd.DataFrame(array1, columns=col_name1)
print(df_array1)


#3개의 컬럼명이 필요함
col_name2=['col1','col2','col3']
#2행 3열 형태의 리스트와 ndarray생성한 뒤 이를 DataFrame으로 변환
list2=[[1,2,3],
       [11,12,13]]
df_list2=pd.DataFrame(list2,columns=col_name2)
df_list2

array2=np.array(list2)
print(array2.shape)
df_array2=pd.DataFrame(array2, columns=col_name2)
df_array2

### 딕셔너리를 DataFrame으로 변환
# key는 컬럼명으로 매핑, Value는 리스트 형(또는 ndarray)
dict={'col1':[1,11],'col2':[2,22],'col3':[3,33]}
df_dict=pd.DataFrame(dict)
df_dict

### DataFrame을 ndarray로 변환  ***
#DataFrame을 ndarray로 변환
array3=df_dict.values
type(array3)
array3.shape
array3
###DataFrame을 리스트와 딕셔너리로 변환
#DataFrame을 리스트로 변환
list3=df_dict.values.tolist()
type(list3)
print(list3)
#DataFrame을 딕셔너리로 변환
dict3=df_dict.to_dict('list')
type(dict3)
dict3


###
### DataFrame의 컬럼 데이터 셋 Access
###
# DataFrame의 컬럼 데이터 세트 생성과 수정은 []연산자를 이용해 쉽게 할 수 있다.

# 새로운 컬럼에 값을 할당하려면 DataFrame[]내에 새로운 컬럼명을 입력하고 값을 할당해주기만 하면 된다.
titanic_df['Age_0']=0
titanic_df.head()

titanic_df['Age_by_10']=titanic_df['Age']*10
titanic_df['Family_No']=titanic_df['SibSp']+titanic_df['Parch']+1
titanic_df.head(3) #기존 컬럼에 값을 업데이트하려면 해당 컬럼에 업데이트값을 그대로 지정

titanic_df['Age_by_10']=titanic_df['Age_by_10']+100
titanic_df.head()


###
### DataFrame 데이터 삭제
###
### axis에 따른 삭제
titanic_drop_df=titanic_df.drop('Age_0',axis=1)
titanic_drop_df.head()

# drop()메소드의 inplace인자의 기본값은 False임.
# 이 경우 drop()호출을 한 DataFrame은 아무런 영향이 없으며
# drop()호출의 결과가 해당 컬럼이 drop된 DataFrame을 반환함
titanic_df.head()
# 여러개의 컬럼들의 삭제는 drop의 인자로 삭제 컬럼들을 리스트로 입력함. inplace=True일 경우
# 호출을 한 DataFrame에 drop이 반영됨. 이때 반환값은 None임

drop_result=titanic_df.drop(['Age_0','Age_by_10','Family_No'], axis=1, inplace=True)
print('inplace=Ture로 drop후 반환된 값:', drop_result) #반환안됨. 이미 원본이 바뀜
titanic_df.head()  

#axis=0일 경우 drop()은 row방향으로 데이터를 삭제함
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 15)
print('#### before axis 0 drop ####')
print(titanic_df.head(3))

titanic_df.drop([0,1,2], axis=0, inplace=True) #인덱스 0 1 2 삭제돼도 3 4 5가 앞으로 땡기진 않음

print('#### after axis 0 drop ####')
print(titanic_df.head(3))


### Index 객체
# 원본 파일 재 로딩 
titanic_df=pd.read_csv('C:/Users/uos/Desktop/data/titanic.csv')
# Index 객체 추출
indexes = titanic_df.index
print(indexes)              #실제로 index 안은 ndarray로 되어있음
# Index 객체를 실제 값 arrray로 변환 
print('Index 객체 array값:\n',indexes.values)
    #index는 1차원 데이터입니다.

print(type(indexes.values))
print(indexes.values.shape)
print(indexes[:5].values)
print(indexes.values[:5])
print(indexes[6])

indexes[0] = 5 # []를 이용해서 임의로 index값 변경은 불가(pk고유성 깨지므로)


# Serise객체는 Index객체를 포함하지만 Series 객체에 연산 함수를 적용할 때 Index는 연산에서 제외됨
# index는 오직 식별용으로만 사용
series_fair = titanic_df['Fare']
print('Fair Series max 값:', series_fair.max())
print('Fair Series sum 값:', series_fair.sum())
print('sum() Fair Series:', sum(series_fair))
print('Fair Series + 3:\n',(series_fair + 3).head(3))

#***
# DataFrame 및  Series에 reset_index() 메서드를 수행하면
# 새롭게 인덱스를 연속 숫자 형으로 할당하며 기존 인덱스는 
# 'index'라는 새로운 컬럼 명으로 추가합니다.
#데이터프레임의 경우
titanic_reset_df = titanic_df.reset_index(inplace=False)
titanic_reset_df.head(3)
titanic_reset_df.shape

#시리즈의 경우
value_counts = titanic_df['Pclass'].value_counts()
print(value_counts)                                                #여기의 index값 필요한데!
print('value_counts 객체 변수 타입:',type(value_counts))

new_value_counts = value_counts.reset_index(inplace=False)
print(new_value_counts)
print('new_value_counts 객체 변수 타입:',type(new_value_counts))

'''

'''
'''


'''
### value_counts()
# 동일한 개별 데이터 값이 몇 건이 있는지 정보를 제공. 즉 개별 데이터값의 분포도를 제공.
# 주의할 점은 value_counts()는 Series객체에서만 호출될 수 있으므로 
# 반드시 DataFrame을 단일 컬럼으로 입력하여 Series로 변환한 뒤 호출
value_counts=titanic_df['Pclass'].value_counts()
value_counts

titanic_pclass=titanic_df['Pclass']
titanic_pclass.head()

value_counts=titanic_df['Pclass'].value_counts()
type(value_counts)
value_counts

### sort_values()      by=정렬컬럼, ascending=True(오름차순)
titanic_df.sort_values(by='Pclass',ascending=False)

titanic_df[['Name','Age']].sort_values(by='Age')
titanic_df[['Name','Age','Pclass']].sort_values(by=['Pclass','Age'])


'''
1. [] :컬럼 기반 필터링 또는 불린 인덱싱 필터링 제공
2. 데이터 셀렉션 및 필터링 - ix, loc, iloc
    ix[] : 명칭 기반과 위치 기반 인덱싱을 함께 제공
    loc[]: 명칭 기반 인덱싱
    iloc[]:위치 기반 인덱싱(행 열 위치값으로 정수가 입력됨) i는 index
3. 불린인덱싱*선*
    위치기반, 명칭기반 사용할 필요 없이 조건식을 []안에 기입하여 간편 필터링
    titanic_boolean=titanic_df[titanic_df['Age']>60]
    
* DataFrame바로뒤 [] 안에 들어갈 수 있는건 컬럼명과 불린인덱싱으로 범위를 좁혀서 코딩하는게 도움됨)
'''
###
### 데이터 셀렉션 및 필터링
###
### DataFrame의 [ ] 연산자
# 넘파이에서 []연산자는 행의 위치, 열의 위치, 슬라이싱 범위 등을 지정해 데이터를 가져올 수 있었다.
# DataFrame에서는 컬럼 명 문자(또는 컬럼 명의 리스트 객체), 또는 인덱스로 변환 가능한 표현식임.
titanic_df=pd.read_csv('C:/Users/uos/Desktop/data/titanic.csv')
print('단일 컬럼 데이터 추출:\n', titanic_df[ 'Pclass' ].head(3))
print('\n여러 컬럼들의 데이터 추출:\n', titanic_df[ ['Survived', 'Pclass'] ].head(3))
#print('[ ] 안에 숫자 index는 KeyError 오류 발생:\n', titanic_df[0]) #할 버릇 하지마

#앞에서 DataFrame의 []내에 숫자 값을 입력할 경우 오류가 발생한다고 했는데, Pandas의 index형태로
#변환 가능한 표현식은 [] 내에 입력 가능
#titanic_df[0:2] 이건 그냥 하지 않는 습관

#
# DataFrame 바로 뒤에 있는 [] 안에 들어갈 수 있는 
# 것은 컬럼명과 불린인덱싱으로 범위를 좁혀서 코딩하는게 도움됨)
#

titanic_df[ titanic_df['Pclass'] == 3].head(3)


### ix연산자
# print('컬럼 위치 기반 인덱싱 데이터 추출:',titanic_df.ix[0,2])
# print('컬럼명 기반 인덱싱 데이터 추출:',titanic_df.ix[0,'Pclass'])

# data = {'Name': ['Chulmin', 'Eunkyung','Jinwoong','Soobeom'],
#         'Year': [2011, 2016, 2015, 2015],
#         'Gender': ['Male', 'Female', 'Male', 'Male']
#        }
# data_df = pd.DataFrame(data, index=['one','two','three','four'])
# data_df


# print("\n ix[0,0]", data_df.ix[0,0])
# print("\n ix['one', 0]", data_df.ix['one',0])
# print("\n ix[3, 'Name']",data_df.ix[3, 'Name'],"\n")

# print("\n ix[0:2, [0,1]]\n", data_df.ix[0:2, [0,1]])
# print("\n ix[0:2, [0:3]]\n", data_df.ix[0:2, 0:3])
# print("\n ix[0:3, ['Name', 'Year']]\n", data_df.ix[0:3, ['Name', 'Year']], "\n")
# print("\n ix[:] \n", data_df.ix[:])
# print("\n ix[:, :] \n", data_df.ix[:, :])

# print("\n ix[data_df.Year >= 2014] \n", data_df.ix[data_df.Year >= 2014])

# 명칭 기반 인덱싱과 위치 기반 인덱싱의 구분
# data_df 를 reset_index() 로 새로운 숫자형 인덱스를 생성
data_df_reset = data_df.reset_index()
data_df_reset = data_df_reset.rename(columns={'index':'old_index'})

# index 값에 1을 더해서 1부터 시작하는 새로운 index값 생성
data_df_reset.index = data_df_reset.index+1
data_df_reset

data_df_reset.ix[0,1]# 오류를 발생합니다. old_index와 index를 동시에 인식하고 심지어 old_index를 인식
data_df_reset.ix[1,1]


### iloc
data_df.head()
data_df.iloc[0, 0]

data_df.iloc[0, 'Name'] # 오류를 발생합니다. 
data_df.iloc['one', 0] # 오류를 발생합니다. 

data_df_reset.iloc[0, 1]


### loc
data_df.loc['one', 'Name']
data_df_reset.head()
data_df_reset.loc[1, 'Name']
data_df_reset.loc[0, 'Name'] # 오류를 발생합니다. 

print('명칭기반 ix slicing\n', data_df.ix['one':'two', 'Name'],'\n')
print('위치기반 iloc slicing\n', data_df.iloc[0:1, 0],'\n')
print('명칭기반 loc slicing\n', data_df.loc['one':'two', 'Name'])

print(data_df_reset.loc[1:2 , 'Name'])


'''
###
### 불린 인덱싱
###
'''
titanic_df = pd.read_csv('titanic_train.csv')
var1=titanic_df['Age'] > 60
type(var1)  #불린값을 가진 시리즈 객체를 dataframe[]안에 넣으면 원하는 데이터를 필터링해서 가져올 수 있음

titanic_boolean = titanic_df[titanic_df['Age'] > 60]
print(type(titanic_boolean))
titanic_boolean

titanic_df[titanic_df['Age'] > 60][['Name','Age']].head(3)
titanic_df[['Name','Age']][titanic_df['Age'] > 60].head(3) #위와 동치

titanic_df.loc[titanic_df['Age'] > 60, ['Name','Age']].head(3) #위와 동치, 행에 조건식, 열에 원하는 열


## * 논리 연산자로 결합된 조건식도 불린 인덱싱으로 적용 가능함
titanic_df[ (titanic_df['Age'] > 60) & (titanic_df['Pclass']==1) & (titanic_df['Sex']=='female')]

## * 조건식은 변수로도 할당 가능. 복잡한 조건식은 변수로 할당하여 가독성을 향상할 수 있음
cond1 = titanic_df['Age'] > 60
cond2 = titanic_df['Pclass']==1
cond3 = titanic_df['Sex']=='female'
titanic_df[ cond1 & cond2 & cond3]
'''
Aggregation 함수
- sum(), max(), min(), count() 등의 함수는 DataFrame/ Series에서 집합(Aggregation)연산을 수행
- DataFrame의 경우 DataFrame에서 바로 aggregation을 호출할 경우 모든 컬럼에 해당 aggregation을 적용

axis에 따른 Aggregation 함수 결과(지정 않으면 axis=0)  
    age    fare
    20     10000   ->10020
    56     20000   ->20056
    12     5000    ->5012
   ->88  ->35000 

titanic_df[['Age','Fare']].sum(axis=0) -> 행 합(컬럼 합이 됨) 88 35000
titanic_df[['Age','Fare']].sum(axis=1) -> 10020 20056 5012

DataFrame Group By
- DataFrame은 Group By 연산을 위해 groupby() 메소드 제공
- groupby 메소드는 by인자로 group by하려는 컬럼명을 입력받으면 DataFrameGroupBy객체를 반환
- 이렇게 반환된 DataFrameGroupBy 객체에 aggregation 함수를 수행
'''
###
### Aggregation 및 GroupBy 함수 적용
###
### Aggregation 함수 
titanic_df.count() #Nan값은 count에서 제외
# 특정 컬럼들로 aggregation 함수 수행 

titanic_df[['Age', 'Fare']].mean()
titanic_df[['Age', 'Fare']].mean(axis=1)
titanic_df[['Age', 'Fare']].sum(axis=0)
titanic_df[['Age', 'Fare']].count()
### groupby(): by인자에 Groupby하고자 하는 컬럼을 입력, 여러개의 컬럼으로 Group by하고자 하면
#[]내에 해당 컬럼명을 입력. DataFrame에 groupby를 호출하면 DataFrameGroupBy 객체를 반환함
titanic_groupby = titanic_df.groupby(by='Pclass')
print(type(titanic_groupby))  
titanic_groupby    # DataFrameGroupBy 객체가 반환됨

#DataFrameGroupBy 객체에 Aggregation함수를 호출하여 Group by 수행
titanic_groupby = titanic_df.groupby('Pclass').count()
titanic_groupby #pclass는 index

type(titanic_groupby)
titanic_groupby.shape  #pclass가 index로 가면서 열-1됨
titanic_groupby.index

titanic_groupby = titanic_df.groupby('Pclass')[['PassengerId', 'Survived']].count()
titanic_groupby

# 더 가볍게도 가능
titanic_groupby1 = titanic_df[['PassengerId', 'Survived','Pclass']].groupby('Pclass').count()
titanic_groupby1

### * Pclass의 값별로 groupby해서 count한 Pclass만 보고싶음. ->둘째줄
titanic_groupby = titanic_df.groupby('Pclass')['Pclass'].count(); titanic_groupby
titanic_df['Pclass'].value_counts()  # 훨씬 쉬운 계산법이 있다

# RDBMS의 group by는 select절에 여러개의 aggregation 함수를 적용할 수 있음
# Select max(Age), min(Age) from titanic_table group by Pclass
# 판다스는 여러개의 aggregation 함수를 적용할 수 있도록 agg()함수를 별도 제공
# *
titanic_df.groupby('Pclass')['Age'].agg([max, min]) #pclass별로 age의 max와 min구하기

#딕셔너리를 이용하여 다양한 aggregation 함수를 적용
agg_format={'Age':'max', 'SibSp':'sum', 'Fare':'mean'} #컬럼별 다양한 agg함수 적용하고픔
titanic_df.groupby('Pclass').agg(agg_format)

'''
결손 데이터 처리하기
- isna() : DataFrame의 isna()메소드는 주어진 칼럼값들이 NaN인지 True/False값을 반환
- fillna() : Missing 데이터를 인자로 주어진 값으로 대체함
'''
###
### 결손 데이터 처리하기
###
###isna()로 결손 데이터 여부 확인
titanic_df.isna().head(3) # 모든 컬럼값들이 NaN인지 True/False 값 반환
titanic_df.isna( ).sum( ) #컬럼별 NaN 개수

### fillna( ) 로 Missing 데이터 대체하기
titanic_df['Cabin'] = titanic_df['Cabin'].fillna('C000') #임의의 선실등급값 할당
titanic_df.head(3)

titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
titanic_df.isna().sum()


### 번외 - 정렬
titanic_sorted = titanic_df.sort_values(by=['Name'])
titanic_sorted.head(3)

titanic_sorted = titanic_df.sort_values(by=['Pclass', 'Name'], ascending=False)
titanic_sorted.head(3)

'''
apply lambda 식으로 데이터 가공
- 판다스는 apply함수에 lambda식을 결합해 DataFrame/Series의 레코드별로 데이터를 가공하는 기능제공.
- 판다스의 경우 컬럼에 일괄적으로 데이터 가공을 하는 것이 속도 면에서 더 빠르나 복잡한 데이터 가공이
- 필요한 경우 어쩔수 없이 apply lambda를 이용
lambda x: x**2     #x:입력인자, x**2:입력 인자를 기반으로 한 계산식, 호출시 계산결과가 반환됨
    ex) titanic_df['Name_len']=titanic_df['Name'].apply(lambda x: len(x))
    
def get_square(a):  # 원랜 두줄
    return a**2
'''
###
### apply lambda 식으로 데이터 가공
###
# 파이썬 lambda식 기본
def get_square(a):
    return a**2
print('3의 제곱은:',get_square(3))

lambda_square = lambda x : x ** 2
print('3의 제곱은:',lambda_square(3))

a=[1,2,3]
squares = map(lambda x : x**2, a)
list(squares)

# 판다스에 apply로 lambda식 사용
titanic_df['Name_len']= titanic_df['Name'].apply(lambda x : len(x)) #네임값 하나씩 들어왔을 때 크기 구해서 대입
titanic_df[['Name','Name_len']].head(3)

# 보통은 elif 간단히 할 때.
titanic_df['Child_Adult'] = titanic_df['Age'].apply(lambda x : 'Child' if x <=15 else 'Adult' )
titanic_df[['Age','Child_Adult']].head(8)
# 불 편 쓰  ->다음 단(함수 먼저 만들고 람다를 씌워)
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : 'Child' if x<=15 else ('Adult' if x <= 60 else 
                                                                                  'Elderly'))
titanic_df['Age_cat'].value_counts()


# 나이에 따라 세분화된 분류를 수행하는 함수 생성. 
def get_category(age):
    cat = ''
    if age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    
    return cat

# lambda 식에 위에서 생성한 get_category( ) 함수를 반환값으로 지정. 
# get_category(X)는 입력값으로 각 행별 ‘Age’값을 받아서 해당하는 cat 반환
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
titanic_df[['Age','Age_cat']].head()
    
'''
2차원 데이터 핸들링을 위해서는 판다스를 사용
판다스는 매우 편리하고 다양한 데이터 처리 API를 제공하지만(조인, 피벗/언피벗/ SQL like API 등),
이를 다 알기에는 많은 시간과 노력이 필요
지금까지 핵심 사항에 집중하고, 데이터 처리를 직접 수행해 보면서 문제에 부딛칠 때마다
판다스의 다양한 API를 찾아서 해결해 가면 판다스에 대한 실력을 더욱 향상시킬 수 있음
->  판다스 레퍼런스 책을 보면서 익히기보다는 지금까지 내용을 중점으로
    기본 익히고 문제 생길때마다 인터넷 검색->API들에 익숙해지게
    
    그렇지만 너무많은 시간을 투자하진 말기
    딱 이정도만 잘 해놓기
    '''
    
    
    

#%%

#%%

# %%