import numpy as np
import pandas as pd
import csv
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font', family='BiauKaiHK')
# Apple LiSung

N, M, NUM = 3, 5, 41

'''
將 Y 設為 val, X = range(N)
求 X, Y 回歸直線斜率判斷反應時間訓練情況
'''

class DATA:
    def __init__(self, N, M, argv):
        self.X =[[float(argv[i*M + j]) for j in range(M)] for i in range(N)]
        self.AVG = [float(sum(self.X[i])/M) for i in range(N)]
        self.D = [self.X[i][M-1] - self.X[i][0] for i in range(N)]
        
        # slope of regression line
        self.MR = [ float(np.polyfit(range(1, M+1), self.X[i], 1)[0]) for i in range(N)]
    
    def p(self):
        print(self.MR)
        
# 讀取 Excel 檔案中的資料
df = pd.read_csv('formed.csv')

df.drop(df.columns[0], axis=1, inplace = True)

for i in range(15):
    df[i] = df[str(i)] 
    df.drop(str(i), axis=1)

data_by_num = []
for num in range(NUM):
    data_by_num.append(DATA(N, M, list([float(df[i][num]) for i in range(N*M)])))


fig, axs = plt.subplots(3)
fig.suptitle('我是圖名')

for i in range(NUM):
    print(i, '\t', data_by_num[i].MR)

X = range(NUM)
axs[0].bar(X, [data_by_num[i].D[0] for i in range(NUM)])
axs[0].bar(X, [data_by_num[i].MR[0] for i in range(NUM)], color='orange')
axs[0].text(-0.2,  8, str(sum(x < 0 for x in [data_by_num[i].D[0] for i in range(NUM)])) + '/41', color = 'blue', size = 13)
axs[0].text(-0.2,  5, str(sum(x < 0 for x in [data_by_num[i].MR[0] for i in range(NUM)])) + '/41', color = 'orange', size = 13)
axs[0].set_xlabel('對照組(座號)')
axs[0].set_ylabel("接到尺的距離 (cm)")

axs[1].bar(X, [data_by_num[i].D[1] for i in range(NUM)])
axs[1].bar(X, [data_by_num[i].MR[1] for i in range(NUM)], color='orange')
axs[1].text(-0.2,  8, str(sum(x < 0 for x in [data_by_num[i].D[1] for i in range(NUM)])) + '/41', color = 'blue', size = 13)
axs[1].text(-0.2,  5, str(sum(x < 0 for x in [data_by_num[i].MR[1] for i in range(NUM)])) + '/41', color = 'orange', size = 13)
axs[1].set_xlabel("99乘法組(座號)")
axs[1].set_ylabel("接到尺的距離 (cm)")

axs[2].bar(X, [data_by_num[i].D[2] for i in range(NUM)])
axs[2].bar(X, [data_by_num[i].MR[2] for i in range(NUM)], color='orange')
axs[2].text(-0.2, 15, str(sum(x < 0 for x in [data_by_num[i].D[2] for i in range(NUM)])) + '/41', color = 'blue', size = 13)
axs[2].text(-0.2,  10, str(sum(x < 0 for x in [data_by_num[i].MR[2] for i in range(NUM)])) + '/41', color = 'orange', size = 13)
axs[2].set_xlabel('拍肩舉手組(座號)')
axs[2].set_ylabel("接到尺的距離 (cm)")



plt.show()

