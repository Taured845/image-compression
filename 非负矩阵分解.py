# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:09:10 2022

@author: 86131
"""

import mnist
import matplotlib.pyplot as plt
import numpy as np
import time


#取出mnist中所有‘8’的图片
(x_train,t_train),(x_test,t_test)=mnist.load_mnist()
X=np.where(t_train==8)[0]
Y=[]
for i in range(len(X)):
    Y.append(x_train[X[i]].reshape(28,28))
#-----------------------------------------#

#压缩
k=5
H=[]
W=[]
Z=[]    #压缩结果
nums=[]    #迭代次数

t1=time.time()
for l in range(len(Y)):
    sum_Y0=np.sum(Y[l],0)   #列和
    sum_Y1=np.sum(Y[l],1)   #行和
    Y_labels0=np.where(sum_Y0!=0)    #非零列
    Y_labels1=np.where(sum_Y1!=0)    #非零行
    Y_new=Y[l][tuple(Y_labels1[0]),:] 
    Y_new=Y_new[:,tuple(Y_labels0[0])]#Y非零部分
    m_=len(Y_labels1[0])    #非零部分行数
    n_=len(Y_labels0[0])    #非零部分列数

    H0=np.random.uniform(0,1,(k,n_))    #H每轮迭代初值
    H1=np.ones((k,n_))    #H每轮迭代后结果
    W0=np.random.uniform(0,1,(m_,k))    #W每轮迭代初值
    W1=np.ones((m_,k))    #W每轮迭代后的结果
    flag=True
    num=0
        
    while (flag==True):
        H1=(H0*(W0.T@Y_new))/(W0.T@W0@H0)
        W1=(W0*(Y_new@H1.T))/(W0@H1@H1.T)
                        
        s_H10=np.linalg.norm(H1-H0)**2
        s_H0=np.linalg.norm(H0)**2
        s_H1=np.linalg.norm(H1)**2
        s_W10=np.linalg.norm(W1-W0)**2
        s_W0=np.linalg.norm(W0)**2
        s_W1=np.linalg.norm(W1)**2
        s_X=np.linalg.norm(Y[l])**2       
        if s_H10<=0.005*s_H0 and s_W10<=0.005*s_W1:
            H11=np.zeros((k,Y_labels0[0][0]))
            H12=np.zeros((k,28-n_-Y_labels0[0][0]))
            H1_=np.concatenate((H11,H1,H12),1)
            
            W11=np.zeros((Y_labels1[0][0],k))
            W12=np.zeros((28-m_-Y_labels1[0][0],k))
            W1_=np.concatenate((W11,W1,W12),0)
            
            H.append(H1_)
            W.append(W1_)
            Z.append(W1_@H1_)
            nums.append(num)
            flag=False
        else:
            H0=H1.copy()
            W0=W1.copy()
            num=num+1
            
t2=time.time()
print(t2-t1)

#画图
plt.figure()
plt.title('compressed image')
plt.imshow(Z[99]) 

plt.figure()
plt.title('original image')
plt.imshow(Y[99])
