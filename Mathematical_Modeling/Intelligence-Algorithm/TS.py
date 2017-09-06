
# coding: utf-8

# In[1]:

import random
import copy

N = 10    # 序列长度
banLen = 500    # 禁忌表长度
ITER = 1000    # 迭代次数
INF = 1e10    # 无穷大

def evaluate(order):
    return 0

def swap(order, i, j):
    t = order[i]
    order[i] = order[j]
    order[j] = t

def updateban(ban, order, t):
    if len(ban) < banLen:
        ban.append(order)
    else:
        ban[t] = order
        
def TS():
    bestorder = range(N)
    random.shuffle(bestorder)
    bestvalue = evaluate(bestorder)
    noworder = copy.copy(bestorder)
    ban = []
    
    for num_iter in range(ITER):
        nextorder = []
        nextvalue = 0
        order = noworder
        
        for i in range(N):
            for j in range(i+1, N):
                swap(order, i, j)
                if not (order in ban):
                    temp = evaluate(order)
                    if temp > nextvalue:
                        nextorder = copy.copy(order)
                        nextvalue = temp;
                swap(order, i, j)
            
        if nextvalue == 0:
            break
        
        if nextvalue > bestvalue:
            bestvalue = nextvalue
            bestorder = nextorder
        updateban(ban, copy.copy(nextorder), num_iter%banLen)
        noworder = nextorder

    return bestvalue, bestorder
            


# In[ ]:



