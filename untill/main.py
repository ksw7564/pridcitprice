from calcpkg.al_analysis import *
from calcpkg.cu_analysis import *
from calcpkg.hr_analysis import *
from calcpkg.ni_analysis import *
from calcpkg.pc_analysis import *
from calcpkg.pp_analysis import *
import datetime
import math


#구리가중치 추가

non_f = pd.read_csv('/data1/KSW/모비스/보고서/비철금속요약의요약.csv',encoding='cp949')
non_f = non_f[['0','1','2','3','4','5']]
Non_f = non_f.values.tolist()

for i in range(len(Non_f)):
    Non_f[i][1]=Non_f[i][1].strip('발행  \n')
    
non_f=pd.DataFrame(Non_f)
non_f[1]= pd.to_datetime(non_f[1], format =  '%Y.%m.%d')
non_f=non_f.sort_values(1)
Non_f=non_f.values.tolist()
Non_f = Non_f[len(Non_f)-1]

t=cu.values.tolist()
t[0]

CU_df = cu_df.values.tolist()
temp = []
temp.append(CU_df[len(CU_df)-1][1])
for i in range(len(t)):
    if math.isnan(t[i][0]) == False:
        temp.append(t[i][0])
    else:
        break

temp1=[]
for i in range(1,len(temp)):
    temp1.append(temp[i]-temp[i-1])

for i in range(len(temp1)):
    if (temp1[i] > 0) & (Non_f[2] == '하방'):
        t[i].append(t[i][0] - (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] > 0) & (Non_f[2] == '하락'):
        t[i].append(t[i][0] - (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] > 0) & (Non_f[2] == '낙폭'):
        t[i].append(t[i][0] - (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] < 0) & (Non_f[2] == '상승'):
        t[i].append(t[i][0] + (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] < 0) & (Non_f[2] == '상승폭'):
        t[i].append(t[i][0] + (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] < 0) & (Non_f[2] == '상방'):
        t[i].append(t[i][0] + (t[i][0]*cu_mape[0]/100))
    elif Non_f == '-':
        t[i].append("값이 비었습니다.")
    else:
        t[i].append(t[i][0])

result_save1 = pd.DataFrame(t)


#알루미늄 가중치

non_f = pd.read_csv('/data1/KSW/모비스/보고서/비철금속요약의요약.csv',encoding='cp949')
non_f = non_f[['0','1','2','3','4','5']]
Non_f = non_f.values.tolist()

for i in range(len(Non_f)):
    Non_f[i][1]=Non_f[i][1].strip('발행  \n')
    
non_f=pd.DataFrame(Non_f)
non_f[1]= pd.to_datetime(non_f[1], format =  '%Y.%m.%d')
non_f=non_f.sort_values(1)
Non_f=non_f.values.tolist()
Non_f = Non_f[len(Non_f)-1]

t=al.values.tolist()


AL_df = al_df.values.tolist()
temp = []
temp.append(AL_df[len(AL_df)-1][1])
for i in range(len(t)):
    if math.isnan(t[i][0]) == False:
        temp.append(t[i][0])
    else:
        break

temp1=[]
for i in range(1,len(temp)):
    temp1.append(temp[i]-temp[i-1])

for i in range(len(temp1)):
    if (temp1[i] > 0) & (Non_f[3] == '하방'):
        t[i].append(t[i][0] - (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] > 0) & (Non_f[3] == '하락'):
        t[i].append(t[i][0] - (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] > 0) & (Non_f[3] == '낙폭'):
        t[i].append(t[i][0] - (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] < 0) & (Non_f[3] == '상승'):
        t[i].append(t[i][0] + (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] < 0) & (Non_f[3] == '상승폭'):
        t[i].append(t[i][0] + (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] < 0) & (Non_f[3] == '상방'):
        t[i].append(t[i][0] + (t[i][0]*cu_mape[0]/100))
    elif Non_f == '-':
        t[i].append("값이 비었습니다.")
    else:
        t[i].append(t[i][0])

result_save2 = pd.DataFrame(t)


#니켈가중치 추가

non_f = pd.read_csv('/data1/KSW/모비스/보고서/비철금속요약의요약.csv',encoding='cp949')
non_f = non_f[['0','1','2','3','4','5']]
Non_f = non_f.values.tolist()

for i in range(len(Non_f)):
    Non_f[i][1]=Non_f[i][1].strip('발행  \n')
    
non_f=pd.DataFrame(Non_f)
non_f[1]= pd.to_datetime(non_f[1], format =  '%Y.%m.%d')
non_f=non_f.sort_values(1)
Non_f=non_f.values.tolist()
Non_f = Non_f[len(Non_f)-1]

t=ni.values.tolist()


NI_df = ni_df.values.tolist()
temp = []
temp.append(NI_df[len(NI_df)-1][1])
for i in range(len(t)):
    if math.isnan(t[i][0]) == False:
        temp.append(t[i][0])
    else:
        break

temp1=[]
for i in range(1,len(temp)):
    temp1.append(temp[i]-temp[i-1])

for i in range(len(temp1)):
    if (temp1[i] > 0) & (Non_f[4] == '하방'):
        t[i].append(t[i][0] - (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] > 0) & (Non_f[4] == '하락'):
        t[i].append(t[i][0] - (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] > 0) & (Non_f[4] == '낙폭'):
        t[i].append(t[i][0] - (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] < 0) & (Non_f[4] == '상승'):
        t[i].append(t[i][0] + (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] < 0) & (Non_f[4] == '상승폭'):
        t[i].append(t[i][0] + (t[i][0]*cu_mape[0]/100))
    elif (temp1[i] < 0) & (Non_f[4] == '상방'):
        t[i].append(t[i][0] + (t[i][0]*cu_mape[0]/100))
    elif Non_f == '-':
        t[i].append("값이 비었습니다.")
    else:
        t[i].append(t[i][0])

result_save3 = pd.DataFrame(t)

#열연가중치 추가

n_f = pd.read_csv('/data1/KSW/모비스/보고서/철강 요약 전망 보고서.csv')
n_f = n_f.T
N_f = n_f.values.tolist()

for i in range(len(n_f)):
    N_f[i][1]=N_f[i][1].strip('발행  \n')
n_f=pd.DataFrame(N_f)
n_f[1]= pd.to_datetime(n_f[1])
n_f=n_f.sort_values(1)
N_f=n_f.values.tolist()
N_f = N_f[len(N_f)-1]

t=hr.values.tolist()

HR_df = hr_df.values.tolist()
temp = []
temp.append(HR_df[len(HR_df)-1][1])
for i in range(len(t)):
    if math.isnan(t[i][0]) == False:
        temp.append(t[i][0])
    else:
        break

temp1=[]
for i in range(1,len(temp)):
    temp1.append(temp[i]-temp[i-1])

for i in range(len(temp1)):
    if (temp1[i] > 0) & (N_f[3] == '하방'):
        t[i].append(t[i][0] - (t[i][0]*hr_mape[0]/100))
    elif (temp1[i] > 0) & (N_f[3] == '하락'):
        t[i].append(t[i][0] - (t[i][0]*hr_mape[0]/100))
    elif (temp1[i] > 0) & (N_f[3] == '낙폭'):
        t[i].append(t[i][0] - (t[i][0]*hr_mape[0]/100))
    elif (temp1[i] < 0) & (N_f[3] == '상승'):
        t[i].append(t[i][0] + (t[i][0]*hr_mape[0]/100))
    elif (temp1[i] < 0) & (N_f[3] == '상승폭'):
        t[i].append(t[i][0] + (t[i][0]*hr_mape[0]/100))
    elif (temp1[i] < 0) & (N_f[3] == '상방'):
        t[i].append(t[i][0] + (t[i][0]*hr_mape[0]/100))
    elif Non_f == '-':
        t[i].append("값이 비었습니다.")
    else:
        t[i].append(t[i][0])

result_save4 = pd.DataFrame(t)

#PP,PC아직

t=pp.values.tolist()
result_save5 = pd.DataFrame(t)

t=pc.values.tolist()
result_save6 = pd.DataFrame(t)

result_save1.to_csv('./cu_', encoding='utf-8')
result_save2.to_csv('./al_', encoding='utf-8')
result_save3.to_csv('./ni_', encoding='utf-8')
result_save4.to_csv('./hr_', encoding='utf-8')
result_save5.to_csv('./pp_', encoding='utf-8')
result_save6.to_csv('./pc_', encoding='utf-8')

print('yes')