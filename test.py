import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math
import platform

matplotlib.rc('font', family=['Microsoft YaHei', 'Apple LiSung'][platform.system()=="Darwin"])

#print(platform.system())

'''
N = group number
M = times per tester do per group
NUM = numbers of testers (?)
PRECISION = float precision
LEAST_COUNT = 0.1 (cm)
'''

def sqrt(x):
    return x**0.5

N, M, NUM, PRECISION, LEAST_COUNT = 3, 5, 41, 3, 0.1


class DATA:
    def __init__(self, sz:int, argv:list):
        self.X = argv
        self.SUM = sum(self.X)
        self.AVG = self.SUM/sz
        
        # ddof = 0/1 -> population/sample deviation
        self.STDEV = np.std(self.X, ddof = 1) 
        
        # Uncertainty A/B/Combination
        self.UA = self.STDEV/sqrt(sz)
        self.UB = LEAST_COUNT/2/sqrt(3)
        self.UC = sqrt(self.UA**2 + self.UB**2)

df = pd.read_csv('formed.csv')
df.drop('num', axis=1, inplace = True)
for i in range(N*M):
    df[i] = df[str(i)] # df[1] = df['1']
    df.drop(str(i), axis=1, inplace=True)


# data_by_num[the number of tester][group_num]
data_by_num = [[DATA(M, df.iloc[num, i*M:(i+1)*M]) for i in range(N)] for num in range(NUM)]

AVG = [sum(data_by_num[num][i].SUM for num in range(NUM)) / (M*NUM)  for i in range(N)]
U = [sqrt(sum(data_by_num[num][i].UC**2 for num in range(NUM))) for i in range(N)]
U = [round(x, PRECISION) for x in U]


X, Y = ["對照組", "99乘法組", "拍肩舉手組"], AVG
plt.bar(X, Y, yerr = U, ecolor = '#00BB00')
plt.title("三種變因的接尺實驗數據長條圖 - 第四組", fontsize = 20)
plt.xlabel("組別", fontsize = 15)
plt.ylabel("接到尺的距離 (cm)", fontsize = 15)
plt.text(-0.2, 30, fr'$\mu_A={U[0]}$' + '\n' + fr'$\mu_B={U[1]}$' + '\n' + fr'$\mu_C={U[2]}$', color = '#00BB00', fontsize = 20)
plt.errorbar(X, Y, yerr=U, ecolor='#00BB00', elinewidth=2, capsize=4, fmt=' ')

plt.show()