import numpy as np
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', family='Apple LiSung')

N, M, NUM = 3, 5, 41

class DATA:
    def __init__(self, N, M, argv):
        self.X =[[argv[i*M + j] for j in range(M)] for i in range(N)]
        self.AVG = [float(sum(self.X[i])/M) for i in range(N)]
        self.STDEV = [float(np.std(np.array(self.X[i]))) for i in range(N)]
        self.VA = [ self.STDEV[i] / ((M-1)**0.5) for i in range(N) ] # M-1 是 mu_A 的那個
        self.VB = [0.1 / ( 2 * ( 3 ** 0.5 ) )] * N
        
        # self.V = [( (self.VA[i]**2) + (self.VB[i]**2) ) for i in range(N)]
        self.V = [( (va**2) + (vb**2) ) for va, vb in zip(self.VA, self.VB) ]
        
        
    def p(self):
        print("AVG =", self.AVG)
        print("STDEV =", self.STDEV)
        print("VA =", self.VA)
        print("V =", self.V)
        
        
# 讀取 Excel 檔案中的資料
df = pd.read_csv('formed.csv')

df.drop(df.columns[0], axis=1, inplace = True)

for i in range(15):
    df[i] = df[str(i)] 
    df.drop(str(i), axis=1)

data_by_num = []
for num in range(NUM):
    data_by_num.append(DATA(N, M, list([float(df[i][num]) for i in range(N*M)])))

data_by_group = DATA(N, M*NUM, list([df[i*M + j][k] for i in range(N) for j in range(M) for k in range(NUM)]))
data_by_group.p()

Z = [ (sum([data_by_num[j].V[i] for j in range(NUM)])) ** 0.5 for i in range(N)]
print("Z =", Z)
print()

print("\n\n")
for i in range(NUM):
    print(i)
    print("VA =", data_by_num[i].VA)
    print("V  =", [data_by_num[i].V[j]**0.5 for j in range(N)])

X = ["對照組", "99乘法組", "拍肩舉手組"]
Y = data_by_group.AVG
V = Z


# V = [round(V[i]*1000)/1000 for i in range(N)]
V = [ round(x, 3) for x in V ]
plt.bar(X, Y, yerr = V, ecolor = '#00BB00')

plt.title("三種變因的接尺實驗數據長條圖 - 第四組", fontsize = 20)
plt.xlabel("組別", fontsize = 15)
plt.ylabel("接到尺的距離 (cm)", fontsize = 15)
plt.text(-0.2, 30, fr'$\mu_A={V[0]}$' + '\n' + fr'$\mu_B={V[1]}$' + '\n' + fr'$\mu_C={V[2]}$', color = '#00BB00', fontsize = 20)
plt.errorbar(X, Y, yerr=Z, ecolor='#00BB00', elinewidth=2, capsize=4, fmt='^')

# plt.savefig("figure.png")
plt.show()


'''
X = range(NUM)
Y = [data_by_num[i].D[0] for i in range(NUM)]
# Z = [ (lambda data_by_num[i].D[0]: 1 if data_by_num[i].D[0] < 0 else 0) for i in range(NUM)]

print(sum(Y)/NUM)

plt.bar(X, Y)
plt.show()
'''