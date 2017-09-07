
# coding: utf-8

# In[2]:

import random

# 变量范围
domain = [(0,100), (0,100)]

def hillclimb(self, domain):
    # 随机产生一个航班序列作为初始种子
    seed = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
    while 1:
        neighbor = []
        # 循环改变解的每一个值，产生一个临近解的列表
        for i in range(len(domain)):
            if seed[i] > domain[i][0]:
                newneighbor = seed[0:i] + [seed[i]-1] + seed[i+1,:]
                neighbor.append(newneighbor)
            if seed[i] < domain[i][1]:
                newneighbor = seed[0:i] + [seed[i]+1] + seed[i+1.:]
                neighbor.append(newneighbor)
                
        # 对所有的临近解计算代价，排序，得到代价最小的解
        neighbor_cost = sorted(
        [(s, self.schedulecost(s)) for s in neighbor], key=lambda x: x[1])
        
        # 如果新的最小代价 > 原种子代价，则跳出循环
        if neighbor_cost[0][1] > self.schedulecost(seed):
            break;
            
        # 新的更小的临近解作为新的种子
        seed = neighbor_cost[0][0]
        print("newseed = ", seed[:], "代价:", self.schedulecost(seed))
        
    # 输出
    self.printschedule(seed)
    print("爬山法得到的解的最小代价是", self.schedulecost(seed))
    return seed


# In[ ]:



