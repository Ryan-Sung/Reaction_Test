import numpy as np
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', family='Apple LiSung')

N, M, NUM = 3, 5, 41

'''
D = 5th.val - 1st.val
see if the react time could be trained
'''

class DATA:
    def __init__(self, N, M, argv):
        '''
        self.length = length
        self.X = val
        self.AVG = sum(val)/length
        self.STDEV = np.std(np.array(val))
        '''
        self.X =[[argv[i*M + j] for j in range(M)] for i in range(N)]
        self.AVG = [float(sum(self.X[i])/M) for i in range(N)]
        self.STDEV = [float(np.std(np.array(self.X[i]))) for i in range(N)]
        self.VA = [ self.STDEV[i] / ((M-1)**0.5) for i in range(N) ] # M-1 是 mu_A 的那個
        self.VB = [0.1 / ( 2 * ( 3 ** 0.5 ) )] * N
        # self.V = [( (self.VA[i]**2) + (self.VB[i]**2) ) ** 0.5  for i in range(N)]
        self.V = [( (self.VA[i]**2) + (self.VB[i]**2) ) for i in range(N)]
        # self.V = [( (self.VA[i]**2) ) for i in range(N)]
        
        # if N == 3 and M == 5:
        #    self.D = [self.X[0][4] - self.X[0][0], self.X[1][4] - self.X[1][0], self.X[2][4] - self.X[2][0]]
        
    def p(self):
        # print(self.X)
        # print(type(self.X))
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



# plt.plot(range(1, 1+M), data_by_num[1].X[0])
# plt.savefig("figure.png")
for i in range(NUM):
    # if data_by_num[i].X[0][4] - data_by_num[i].X[0][0] < 0:
    plt.plot(range(1, 1+M), data_by_num[i].X[0])

plt.xticks(range(1, 1+M), range(1, 1+M))
plt.show()


'''
X = range(NUM)
Y = [data_by_num[i].D[0] for i in range(NUM)]
# Z = [ (lambda data_by_num[i].D[0]: 1 if data_by_num[i].D[0] < 0 else 0) for i in range(NUM)]

print(sum(Y)/NUM)

plt.bar(X, Y)
plt.show()
'''