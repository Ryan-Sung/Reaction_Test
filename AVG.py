import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', family='Apple LiSung')

N, M, NUM, PRECISION = 3, 5, 41, 3

class DATA:
    def __init__(self, N:int, M:int, argv:list):
        self.X =[[argv[i*M + j] for j in range(M)] for i in range(N)]
        
        self.AVG = [sum(X)/M for X in self.X]
        
        # ddof = 1 -> sample deviation not population one
        self.STDEV = [np.std(X, ddof=1) for X in self.X]
        
        # type A/B uncertainty
        self.VA = [ stdev / (M**0.5) for stdev in self.STDEV ]
        self.VB = [0.1 / ( 2 * ( 3 ** 0.5 ) )] * N
        
        # combination uncertainty
        self.V = [( (va**2) + (vb**2) ) for va, vb in zip(self.VA, self.VB) ]
        
        
    def print(self):
        print("AVG =", self.AVG)
        print("STDEV =", self.STDEV)
        print("VA =", self.VA)
        print("V =", self.V)
        
        
# 讀取 Excel 檔案中的資料
df = pd.read_csv('formed.csv')

df.drop(df.columns[0], axis=1, inplace = True)

for i in range(15):
    df[i] = df[str(i)] # df[1] = df['1']
    df.drop(str(i), axis=1, inplace=True)
    
    
data_by_num = [DATA(N, M, df.iloc[[num]]) for num in range(NUM)]

data_by_group = DATA(N, M*NUM, list([df[i*M + j][k] for i in range(N) for j in range(M) for k in range(NUM)]))

V = [ (sum([data_by_num[j].V[i] for j in range(NUM)])) ** 0.5 for i in range(N)]
V = [ round(x, PRECISION) for x in V ]

X = ["對照組", "99乘法組", "拍肩舉手組"]
Y = data_by_group.AVG

plt.bar(X, Y, yerr = V, ecolor = '#00BB00')

plt.title("三種變因的接尺實驗數據長條圖 - 第四組", fontsize = 20)
plt.xlabel("組別", fontsize = 15)
plt.ylabel("接到尺的距離 (cm)", fontsize = 15)
plt.text(-0.2, 30, fr'$\mu_A={V[0]}$' + '\n' + fr'$\mu_B={V[1]}$' + '\n' + fr'$\mu_C={V[2]}$', color = '#00BB00', fontsize = 20)
plt.errorbar(X, Y, yerr=V, ecolor='#00BB00', elinewidth=2, capsize=4, fmt=' ')

plt.show()
