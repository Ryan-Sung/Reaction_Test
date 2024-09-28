import matplotlib.font_manager
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', family='Microsoft YaHei')
        
N, M, NUM = 3, 5, 41
        
# 讀取 Excel 檔案中的資料
df = pd.read_csv('formed.csv')

df.drop(df.columns[0], axis=1, inplace = True)

group = [[[]]for a in range(3)] 
for i in df:
    n = 0
    for j in df[i]:
        i = int(i)
        if len(group[0]) - 1 < n: 
            for k in group : k.append([])
        group[int(i/M)][n].append(j)
        n+=1

def uC(samples:list[list]):
    sum = 0
    pow_ub = 0.01/12 # pow(0.1/2/sqrt(3)) = 0.01 / (4*3)
    for i in samples:
        sum += pow(np.std(i, ddof=1), 2)/len(i) + pow_ub
    print(sum)
    return np.sqrt(sum)

X = ["對照組", "99乘法組", "拍肩舉手組"]
Y = [np.average(i) for i in group]
V = [round(uC(i), 3) for i in group]

plt.bar(X, Y, yerr = V, ecolor = '#00BB00')

plt.title("三種變因的接尺實驗數據長條圖 - 第四組", fontsize = 20)
plt.xlabel("組別", fontsize = 15)
plt.ylabel("接到尺的距離 (cm)", fontsize = 15)
plt.text(-0.2, 30, fr'$\mu_A={V[0]}$' + '\n' + fr'$\mu_B={V[1]}$' + '\n' + fr'$\mu_C={V[2]}$', color = '#00BB00', fontsize = 20)
plt.errorbar(X, Y, yerr=V, ecolor='#00BB00', elinewidth=2, capsize=4, fmt=' ')

# plt.savefig("figure.png")
plt.show()