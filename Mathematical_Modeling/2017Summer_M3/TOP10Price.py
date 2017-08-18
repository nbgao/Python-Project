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

import xlwt
book = Workbook()
sheet = book.add_sheet('Price')

for i in range(len(TOP10)):
    df = ts.get_hist_data(TOP10[i],start_date,end_date)
    price = df[u'close'][::-1]
    sheet.write(i,0,TOP10[i])
    for j in range(len(price)):
        sheet.write(i,j+1,price[j])
        
    price.plot(legend = True , figsize = (10,4))
    plt.title(TOP10[i])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

book.save('TOP10_Price.xls')