
# coding: utf-8

# In[1]:

import random
import math

# 变量范围
domain = [(0,100), (0,100)]

def annealingoptimize(self, domain, T=10000.0, cool=0.98, step=1):
    # 随即初始化值
    vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
    
    while T > 0.1:
        # 选择一个索引值
        i = random.randint(0, len(domain)-1)
        # 选择一个改变索引值的方向
        c = random.randint(-step, step)  # -1,0,1
        # 构造新的解
        vec_new = vec[:]
        vec_new[i] += c
        if (vec_new[i] < domain[i][0]):    # 判断越界情况
            vec_new[i] = domain[i][0]
        if (vec_new[i] > domain[i][1]):
            vec_new[i] = domain[i][1]
            
        # 计算当前成本和新的成本
        cost1 = self.schedulecost(vec)
        cost2 = self.sechdulecost(vec_new)
        
        # 判断新的解是否由于原始解 或者 算法将以一定概率接受较差的解
        if(cost2 < cost1 or random.random()<math.exp(-(cost2-cost1)/T)):
            vec = vec_new
     
        T *= cool    # 温度冷却
        print(vec_new[:], '代价:', self.schedulecost(vec_new))
        
    self.printschedule(vec)
    print('模拟退火算法得到的最小代价是:', self.schedulecost(vec))
    
    return vec


# In[ ]:



