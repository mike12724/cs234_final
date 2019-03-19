import numpy as np
import pandas as pd

if __name__ == '__main__':
    #import data
    f = 'warfarin_S1f.xlsx'
    xl = pd.ExcelFile(f)
    df1 = xl.parse('Sheet1')
    patients = df1.values

    df1 = xl.parse('Sheet2')
    labels = df1.values
    
    s1f_weights = np.array([-0.2546,0.0118,0.0134,-0.6752,0.4060,0.0443,1.2799,-0.5695,4.0376])
    #don't forget to square and divide by 7!

    num_patients = len(patients[:,0])
    num_valid = 0
    five_mg_correct = 0
    s1f_correct = 0
    for i in range(num_patients):
        patient = patients[i,:]
        if patient[8] == 0:
            continue
        label = labels[i]
        dosage = np.inner(s1f_weights, patient)
        dosage = (dosage**2)/7
        if dosage >= 7.0 and label >= 7.0:
            s1f_correct += 1
        elif dosage < 3.0 and label < 3.0:
            s1f_correct += 1
        elif abs(dosage - 5.0) <= 2.0 and abs(label - 5.0) <= 2.0:
            s1f_correct += 1

        if abs(label - 5.0) <= 2.0:
            five_mg_correct += 1

        num_valid += 1

    five_mg_correct = float(five_mg_correct)/num_valid
    s1f_correct = float(s1f_correct)/num_valid
    print five_mg_correct
    print s1f_correct
        
    
    

