import tushare as ts
import pandas as pd
import numpy as np
import scipy
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import scipy.stats as scs
sns.set(color_codes=True,style='whitegrid')
warnings.filterwarnings('ignore')

start_date = '2016-12-1'
end_date = '2017-7-5'
TOP10 = ['601398','601857','601288','601988','600028','600519','601628','601318','600036','601088']

df = ts.get_hist_data(TOP10[0],start_date,end_date)
price = df[u'close'][::-1]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
price.plot(legend = True , figsize = (10,4))
plt.show()


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA

#原自相关图和偏自相关图
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
plot_acf(price,ax=ax1).show()       #ACF
plot_pacf(price,ax=ax2).show()      #PACF

#平稳性检验
print (u'原始序列的ADF检验结果为：',ADF(price))    
#返回值以此为ADF、P-Value、usedlag、nobs、critical values、icbest、regresults、resstore

#一阶差分后的结果
D_price = price.diff().dropna()
D_price.plot(figsize = (10,4))      #差分后股价时间序列图

ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2)
plot_acf(D_price,ax=ax1).show()     #一阶差分ACF
plot_pacf(D_price,ax=ax2).show()    #一阶差分PACF

plot_pacf(D_price).show()           #一阶差分ADF检验
print (u'差分序列的ADF检验结果为：',ADF(D_price))

#白噪声检验
statics,p_value = acorr_ljungbox(D_price,lags = 1)
print (u'差分序列的白噪声检验结果为： Q-statics = %s    p-value = %s'%(statics,p_value))  #返回统计量和p值

#ARIMA差分自回归滑动平均模型
#定阶
'''
pmax = int(len(D_price)/10)
qmax = int(len(D_price)/10)
BIC_matrix = []
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
        try:
            tmp.append(ARIMA(price,(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    BIC_matrix.append(tmp)
        
BIC_matrix = pd.DataFrame(BIC_matrix)
'''
p,q = BIC_matrix.stack().idxmin()
print(u'BIC最小值的p值和q值为:%s、%s'%(p,q))
model = ARIMA(price,(p,1,q)).fit()  #建立ARIMA(0,1,1)模型
model.summary()
#model.forecast(5)