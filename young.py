import matplotlib.font_manager
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
    
print(int(True))

# matplotlib.rc('font', family='Microsoft YaHei')
matplotlib.rc('font', family='Apple LiSung')


N, M, NUM = 3, 5, 41
        
# 讀取 Excel 檔案中的資料
df = pd.read_csv('formed.csv')

df.drop(df.columns[0], axis=1, inplace = True)

group = [[] for i in range(3)]
for i in df:
    n = 0
    for j in df[i]:
        if int(int(i)/5) == 0:
            if len(group[0]) - 1 < n:
                group[0].append([])
                group[1].append([])
                group[2].append([])
            group[0][n].append(j)
        elif int(int(i)/5) == 1:
            group[1][n].append(j)
        else:
            group[2][n].append(j)
        n+=1
# print(group)
def uC(samples:list[list]):
    sum = 0
    pow_ub = 0.01/12 # pow(0.1/2/sqrt(3)) = 0.01 / (4*3)
    for i in samples:
        sum += pow(np.std(i, ddof=1), 2)/n + pow_ub
    return np.sqrt(sum)
    
    

X = ["對照組", "99乘法組", "拍肩舉手組"]
Y = [np.average(i) for i in group]
V = [uC(i) for i in group]


# V = [round(V[i]*1000)/1000 for i in range(N)]
V = [ round(x, 3) for x in V ]
plt.bar(X, Y, yerr = V, ecolor = '#00BB00')

plt.title("三種變因的接尺實驗數據長條圖 - 第四組", fontsize = 20)
plt.xlabel("組別", fontsize = 15)
plt.ylabel("接到尺的距離 (cm)", fontsize = 15)
plt.text(-0.2, 30, fr'$\mu_A={V[0]}$' + '\n' + fr'$\mu_B={V[1]}$' + '\n' + fr'$\mu_C={V[2]}$', color = '#00BB00', fontsize = 20)
plt.errorbar(X, Y, yerr=V, ecolor='#00BB00', elinewidth=2, capsize=4, fmt=' ')

# plt.savefig("figure.png")
plt.show()