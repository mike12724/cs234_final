import numpy as np
import pandas as pd
from collections import defaultdict as dd

f = 'warfarin_orig.xlsx'
xl = pd.ExcelFile(f)
df1 = xl.parse('warfarin')
vals = df1.values
df1 = xl.parse('Sheet1')
labels_out = df1.values

final_out = np.zeros((len(vals[:,0]),9)) #LAST NUMBER NEEDS TO BE A 1!
for j in range(len(vals[:,0])): #j = row number
    if j % 1000 == 0:
        print j
    if type(vals[j,3]) is not int:
        continue
    if type(vals[j,4]) is unicode:
        continue
    if type(vals[j,5]) is unicode:
        continue
    if type(vals[j,21]) is unicode:
        continue
    patient_arr = np.zeros(9)
    patient_arr[8] = 1.0
    patient_arr[0] = vals[j,3]/10 if type(vals[j,3]) is int else 0
    patient_arr[1] = vals[j,4] if type(vals[j,4]) is not unicode else 0
    patient_arr[2] = vals[j,5] if type(vals[j,5]) is not unicode else 0
    patient_arr[3] = 1.0 if 'asian' in vals[j,1].lower() else 0
    patient_arr[4] = 1.0 if 'black' in vals[j,1].lower() else 0
    patient_arr[5] = 1.0 if 'unknown' in vals[j,1].lower() else 0
    break_now = False
    for q in range(3):
        if vals[j,22+q] == 1:
            patient_arr[6] = 1.0
        if type(vals[j,22+q]) is unicode:
            break_now = True
            break
    if break_now: continue
    
    patient_arr[7] = 1.0 if vals[j,21] == 1.0 else 0
    final_out[j,:] = patient_arr

df2 = pd.DataFrame(np.array(final_out))
df2.to_excel('warfarin_S1f.xlsx')
df2 = pd.DataFrame(np.array(labels_out))
df2.to_excel('warfarin_S1ssf.xlsx')

                
