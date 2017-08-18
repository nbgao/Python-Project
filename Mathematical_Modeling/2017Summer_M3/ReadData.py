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

import xlrd
from xlwt import Workbook
book1 = xlrd.open_workbook('Test.xls')
table = book1.sheets()[0]  

book2 = Workbook()
sheet1 = book2.add_sheet('W_Test')

for i in range(1,table.ncols-1):
    Y = table.col_values(i)
    Y = Y[1:]
    W = []
    w = scipy.stats.shapiro(Y)
    W.append(w)
    sheet1.write(i,0,w[0])
    sheet1.write(i,1,w[1])

book2.save('W_Test.xls')
'''
T = []
for i in range(1,table.ncols-1):
    t = table.col_values(i)
    t = t[1:]
    T.append(t)
'''
    
    
    