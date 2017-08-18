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
colors = ['blue','green','red','cyan','magenta','yellow']

import xlrd
book1 = xlrd.open_workbook('Test.xls')
table = book1.sheets()[0]

for i in range(1,10):
    Y = table.col_values(65+i)
    Y = Y[1:]
    sub = plt.subplot(330+i)
    #sub,axes = plt.subplots(nrows = 4 , ncols = 4)
    sub.set_title('Y'+str(65+i))
    #sub.set_xlabel('Y')
    plt.axis([-0.1,0.1,0,50])
    plt.sca(sub)
    sns.axes_style('darkgrid')
    sns.distplot(Y , bins = 50 , kde = True , rug = False , color = colors[i%3])
'''with sns.axes_style('darkgrid'):
    #sub = plt.subplot(220+i)
    #plt.sca(sub)
    sns.distplot(Y, bins=50, kde=True, rug=False)
'''
plt.show()
