# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:47:11 2022

@author: 86131
"""

import mnist
import matplotlib.pyplot as plt
import numpy as np


#取出mnist中所有‘8’的图片
(x_train,t_train),(x_test,t_test)=mnist.load_mnist()
X=np.where(t_train==8)[0]
Y=[]
for i in range(len(X)):
    Y.append(x_train[X[i]].reshape(28,28))
#-----------------------------------------#


#压缩
k1=[] #方法1逼近矩阵的秩
k2=[] #方法2逼近矩阵的秩
Z1=[] #方法1逼近矩阵
Z2=[] #方法2逼近矩阵

for h in range(len(Y)):
    [U,S,V]=np.linalg.svd(Y[h])
    
    #1.选适当的秩，使得逼近矩阵F范数(N)与原矩阵F范数(M)之比大于95%
    M=float(np.dot(S,S))
    N=0
    for i in range(len(S)):
        N=N+S[i]*S[i]
        p=N/M
        if p>0.95:
            k1.append(i)
            break
    #构造逼近矩阵
    Z1.append((U[:,0:k1[h]]*S[0:k1[h]])@V[0:k1[h],:])
    #-----------------------------------------#
    
    #2.通过寻找gap的方式做逼近矩阵
    R=0
    for i in range(len(S)-2):
        r_prev=S[i]/S[i+1]
        R=R+r_prev
        r_average=R/(i+1)
        r_new=S[i+1]/S[i+2]
        if r_new/r_average>100:
            k2.append(i+1)
            break
    #构造逼近矩阵
    Z2.append((U[:,0:k2[h]]*S[0:k2[h]])@V[0:k2[h],:])
    #-----------------------------------------#
#-----------------------------------------#


#画图
plt.figure()
plt.title('original image')
plt.imshow(Y[99])

plt.figure()
plt.title('first method')
plt.imshow(Z1[99])

plt.figure()
plt.title('twice method')
plt.imshow(Z2[99])                 

