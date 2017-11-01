
# coding: utf-8

# In[8]:

# 输入状态个数
N =int(input())

S = []
for i in range(N):
    S.append(str(i))

S


# In[2]:

# 输入状态转换表 s c e
'''
    s: 起始状态
    c: 转移字符
    e: 到达状态
'''
Table = []
M = int(input())
for i in range(M):
    Table.append([])
    t = input()
    Table[len(Table)-1] = t.split(' ')

Table


# In[5]:

# 接受(结束)状态表
E = []
t = input()
E = t.split(' ')


# In[6]:

E


# In[10]:

# ε-closure闭包算法
def Closure(a, T):
    b = a
    while 1:
        s = []
        for i in a:
            for j in range(len(T)):
                if(i==T[j][0] and T[j][1]=='~'):
                    s.append(T[j][2])
        if(len(s)==0):
            break;
        else:
            for i in s:
                b.append(i)
                a = s
    b = sorted(b)
    return b    


# In[11]:

def Edge(a, c, T):
    s = []
    for i in a:
        for j in range(len(T)):
            if(i==T[j][0] and T[j][1]==c):
                s.append(T[j][2])
    s = sorted(s)
    return s


# In[50]:

S1 = []
S2 = []
S1.append(Closure(['0'], Table))
S2.append(0)

while(S2.pop(len(S2)-1)==0):
    S2.append(0)
    for i in range(len(S1)):
        if(S2[i]==0):
            S2[i] = 1
            for c in ['0','1']:
                s = Edge(S1[i], c, Table)
                print(s)  ###
                if(s==[]):
                    break
                s = Closure(s, Table)
                print(s)   ###
                if(s==[]):
                    break
                    
                flag = 0        # flag统计S1中是否有s集合
                for m in S1:
                    if(s==m):
                        flag = 1
                        break
                        
                if(flag==0):
                    S1.append(s)
                    S2.append(0)
                    
print('输出NFA构造的子集:')
print(S1)


print('输出DFA:')
print('S', end = ' ')
for x in range(len(['0','1'])):
    print(x, end = ' ')
print('\n')

# 输出DFA
for i in range(len(S1)):
    print(i, end = ' ')
    for c in ['0','1']:
        s = Edge(S1[i], c, Table)
        if(s==[]):
            print(s, end=' ')
            continue
        s = Closure(s, Table)
        if(s==[]):
            print(s, end=' ')
            continue
        for k in range(len(S1)):
            if(S1[k]==s):
                print(k, end=' ')
                break
    y = 0
    for m in E:
        if m in S1[i]:
            y = y + 1
    
    if(y==0):
        print(0)
    else:
        print(1)
    print('\n')
    
            


# In[43]:

c = ['0', '1']
c[0]


# In[46]:

s1 = Edge(S1[0], c[0], Table)
s1


# In[47]:

s2 = Closure(s1, Table)
s2


# In[21]:

s = ['9', '14', '13', '9', '5', '13', '8', '6', '7', '5']


# In[30]:

s1 = list(map(int, s))
s1 = sorted(s1)
s1


# In[61]:

# int 列表转 set
s2 = set(map(int, s))
s2


# In[57]:

for i in range(len(['0', '1'])):
    print(i)


# In[62]:

for i in s2:
    print(i)


# In[26]:

for c in ['0','1']:
    print(c)


# In[ ]:



