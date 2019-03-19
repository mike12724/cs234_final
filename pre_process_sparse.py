import numpy as np
import pandas as pd
from collections import defaultdict as dd

INR_SPLIT = 0.2
HEIGHT_SPLIT = 10
WEIGHT_SPLIT = 10

HEIGHT_COL = 5-1
WEIGHT_COL = 6-1
IND_COL = 7-1 #indication column
INR_COL = 12-1

SPECIAL_COLS = [HEIGHT_COL,WEIGHT_COL, INR_COL]
SPLIT_PER_COL = np.array([HEIGHT_SPLIT, WEIGHT_SPLIT, INR_SPLIT])

f = 'warfarin_imputed.xlsx'
xl = pd.ExcelFile(f)
df1 = xl.parse('Sheet1')
vals = df1.values
uniques = [np.unique(np.sort(vals[:,i])) for i in SPECIAL_COLS]
col_mins = np.array([uniques[i][0] for i in range(3)])
col_maxs = np.array([uniques[i][-2] for i in range(3)])

for k in range(3):
    i = SPECIAL_COLS[k]
    a = [vals[j,i] for j in range(vals.shape[0]) if type(vals[j,i]) is not unicode]
    m = np.mean(np.array(a))
    for j in range(vals.shape[0]):
        vals[j,i] = vals[j,i]-m if type(vals[j,i]) is not unicode else vals[j,i]
        
    a = [vals[j,i] for j in range(vals.shape[0]) if type(vals[j,i]) is not unicode]
    n = np.linalg.norm(np.array(a))
    for j in range(vals.shape[0]):
        vals[j,i] = vals[j,i]/n if type(vals[j,i]) is not unicode else vals[j,i]
    


#num_buckets_per_col = np.ceil((col_maxs-col_mins)/SPLIT_PER_COL)

#first row = male/female, so not_num_to_num[0] = ['male','female']
#Note: May want to use support > 0.1 for filtering
not_num_to_num = [[] for i in range(len(vals[0,:]))]
com_list = [] #list of possible comorbidities

#populates not_num_to_num
for j in range(len(vals[:,0])): #j = row number
    if j % 1000 == 0:
        print j
    for i in range(len(vals[0])): #i = col number
        if type(vals[j,i]) is unicode:
            vals[j,i] = vals[j,i].lower()
        val = vals[j,i]
        if i in SPECIAL_COLS:
            continue
        if i == IND_COL:
            continue
        else:
            if not val in not_num_to_num[i]:
                not_num_to_num[i].append(val)

#9 indicators (1 for NA), 3 special columns each containing an NA
total_num_features = sum([len(x) for x in not_num_to_num]) +  \
                     9 + 6
binary_features = np.zeros((vals.shape[0], total_num_features))
#create binary feature vectors using dictionary

for j in range(len(vals[:,0])): #j = row number
    if j % 1000 == 0:
        print j
    patient_arr = []
    for i in range(len(vals[0])): #i = col number
        val = vals[j,i]
        a = np.zeros(len(not_num_to_num[i]))
        if i == IND_COL:
            a = np.zeros(9)
            if val != 'not applicable':
                if type(val) != int:
                    individuals = val.split(';')
                    individuals = [int(x) for x in individuals]
                    for indication in individuals:
                        a[indication-1] = 1.0
                else:
                    a[val-1] = 1.0               
            else:
                a[8] = 1.0

        elif i in SPECIAL_COLS:
            k = SPECIAL_COLS.index(i)
            a = np.zeros(2)
            if type(val) is unicode:
                a[1] = 1.0
            else:
                a[0] = val

        else:
            idx = not_num_to_num[i].index(val)
            a[idx] = 1.0
            
        patient_arr.append(a)
        
    binary_features[j,:] = np.concatenate(patient_arr)

df2 = pd.DataFrame(binary_features)
df2.to_excel('warfarin_imputed_supersparse.xlsx')    
    
