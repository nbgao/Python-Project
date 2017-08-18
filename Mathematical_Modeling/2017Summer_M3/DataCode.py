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


def string2date(string):
    return datetime.strptime(string,'%Y-%m-%d')


def substractDate(dateStart,dateEnd):
    date1 = string2date(dateStart)
    date2 = string2date(dateEnd)
    return (date2 - date1).days


from xlwt import Workbook
book = Workbook()
sheet1 = book.add_sheet('Data1')

start_date = '2016-12-1'
end_date = '2017-5-1'

count_complete = 0
count_lack = 0
count_null = 0
Max = 3000
number = 0

while number <= Max:
    stock = str(600000 + number)
    df = ts.get_hist_data(stock , start = start_date , end = end_date)
    try:
        if(len(df) == 136):
            count_complete += 1
            sheet1.write(count_complete,0,stock)
            price = df[u'close'][::-1]
            price1 = np.log(price)
            price2 = np.log(price.shift(1))
            Y = price1 - price2
            Y =  Y.dropna()
            list = []
            for i in range(len(Y)):
               list.append(Y[i])
               sheet1.write(count_complete,i+1,Y[i])
            '''with sns.axes_style('darkgrid'): 
               plt.subplot(111)
               sns.distplot(jiegou, bins=50, kde=True, rug=False)'''
            print (stock,len(list))
        else:
            count_lack += 1
            print (stock,'lack')
    except:
        count_null += 1
        print (stock,'null')
    number += 1
    book.save('excel_test.xls')

#plt.show()

print('count_complete:',count_complete)
print('count_lack:',count_lack)
print('count_null:',count_null)

print(list)
print(scs.kstest(list,'norm'))
print(scs.normaltest(list)[1])
print(scipy.stats.shapiro(list))
