
# coding: utf-8

# In[7]:

import random

'''
popsize-种群数量 
step-变异改变的大小
mutprob-交叉和变异的比例
elite-直接遗传的比例
maxiter-最大迭代次数
'''

def geneticopimize(self, domain, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100):
    
    # Mutate
    def mutate(vec):
        i = random.randint(0,len(domain)-1)
        if(random.random()<0.5 and vec[i]>domai[i][0]):
            return vec[0:i] + [vec[i] - step] + vec[i+1:]
        elif(vec[i] < domain[i][1]):
            return vec[0:i] + [vec[i] + step] + vec[i+1:]
        
    # Crossover
    def cross(r1, r2):
        i = random.randint(0, len(domain)-1)
        return r1[0:i] + re[1:]
    
    # Initpop
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        pop.append(vec)
    
    # 每一代中有多少胜出者
    toplite = int(elite * popsize)
    
    # Main
    for i in range(maxiter):
        scores = [(self.schedulecost(v), v) for v in pop]
        scores.sort()
        ranked = [v for (s,v) in scores]  # 按代价由小到大排序
        
        # 优质解遗传到下一代
        pop = ranked[0:topelite]
        
        # 如果当前种群数量小于既定数量，则添加变异和交叉遗传
        while(len(pop) < popsize):
            # 随机数小于 mutprob 则变异，否则交叉
            if(random.random() < mutprob):  # mutprob 控制交叉和变异比例
                # 选择一个个体
                c = random.randint(0, topelite)
                # Mutate
                pop.append(mutate(ranked[c]))
            else:
                # 随机选择两个个体进行交叉
                c1 = random.randint(0, topelite)
                c2 = random.randint(0, topelite)
                pop.append(crossover(ranked[c1], ranked[c2]))
        #输出当前种群中代价最小的解
        print(scores[0][1], '代价:', scores[0][0])
    self.printschedule(scores[0][1])
    print('遗传算法求得最小代价:', scores[0][0])
    
    return scores[0][1]


# In[ ]:



