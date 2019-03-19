import numpy as np
import pandas as pd
import random

#create TreatmentPlan class
#methods: take in patient data, output dose recommendation
class Treatment:
    def __init__(self, alpha):
        self.alpha = alpha
        self.A = None #list of 3 matrices
        self.b = None #list of 3 vectors

    def calculate_dosage(self, data):
        action = 0
        if self.A is None:
            self.A = [0,0,0]
            self.b = [0,0,0]
            for i in range(3):
                self.A[i] = np.eye(len(data))
                self.b[i] = np.zeros(len(data))
            action = random.randint(0,2) #0 = < 3mg; 2 = > 7mg
        else:
            theta = [0,0,0]
            p = np.zeros(3)
            for i in range(3):
                inverse = self.A[i]
                theta[i] = inverse.dot(self.b[i])
                p[i] = np.dot(theta[i],data) + self.alpha*np.sqrt(data.dot(inverse).dot(data))
            action = np.random.choice(np.flatnonzero(p == p.max()))
        return action

    def update_params(self, data, action, reward):
        #sherman-morrison formula
        self.A[action] -= (self.A[action].dot(np.outer(data,data)).dot(self.A[action]))/(1+data.dot(self.A[action]).dot(data))        
        self.b[action] += reward*data
        

if __name__ == '__main__':
    #import data
    print "Importing Data"
    f = 'warfarin_binary_supersparse.xlsx'
    xl = pd.ExcelFile(f)
    df1 = xl.parse('Sheet1')
    patients = df1.values

    f = 'warfarin_orig.xlsx'
    xl = pd.ExcelFile(f)
    df1 = xl.parse('Sheet2')
    labels = df1.values

    f = 'warfarin_S1f.xlsx'
    xl = pd.ExcelFile(f)
    df1 = xl.parse('Sheet1')
    warf_patients = df1.values

    print "Starting runs"
   
    #tuneable parameters; confidence = 1 - delta
    delta = 0.05
    NUM_RUNS = 10
    alpha = 1 + np.sqrt(np.log(2.0/delta)/2)

    #initialize storage for excel output at the end
    regret_per_run = [[[] for i in range(NUM_RUNS)] for j in range(3)] #regret for 3 different algs
    frac_per_run = [[[] for i in range(NUM_RUNS)] for j in range(3)] #correct fraction for 3 different algs
    action_chosen = [[] for i in range(NUM_RUNS)]
    is_warf = [[] for i in range(NUM_RUNS)]

    #Initialize weights for pharmacogenetic algorithm
    p_weights = np.array([-0.2546,0.0118,0.0134,-0.6752,0.4060,0.0443,1.2799,-0.5695,4.0376])

    c = list(zip(patients,labels,warf_patients))
    for k in range(NUM_RUNS):
        print "Run: " + str(k)
        #initialize treatment algorithm
        treat_bandit = Treatment(alpha)

        #initialize regret list of 3 lists of regret/patient
        regret = [[],[],[]]
        frac_correct = [[],[],[]]
        num_correct = [0,0,0] #5mg, warfarin alg, bandits
        
        #Shuffle data
        random.shuffle(c)
        p,l,w = zip(*c)

        p = np.array(p)
        l = np.concatenate(l)
        w = np.array(w)
        #For each patient, run all 3 algorithms
        for i in range(len(p[:,0])):
            if i % 1000 == 0:
                print "Patient " + str(i)      
            patient = p[i,:]

            #calculate dosage using the 3 algorithms
            
            warf = w[i,:]
            warf_out = (p_weights.dot(warf)**2)/7.0
            warf_action = 0
            if warf_out > 7.0:
                warf_action = 2
            elif warf_out >= 3.0:
                warf_action = 1

            bandit_action = treat_bandit.calculate_dosage(patient)
            action_chosen[k].append(bandit_action)

            #calculate reward for 5mg
            if abs(l[i] - 5.0) <= 2.0: 
                num_correct[0] += 1
                regret[0].append(0)
            else:
                regret[0].append(1)

            #calculate reward for warfarin 
            #out = 0 => not enough patient info
            if warf_out != 0:
                reg = 1
                if l[i] > 7.0 and warf_action == 2:
                    reg = 0
                elif abs(l[i]- 5.0) <= 2.0 and warf_action == 1:
                    reg = 0
                elif l[i] < 3.0 and warf_action == 0:
                    reg = 0
                regret[1].append(reg)
                num_correct[1] += (1-reg)
                is_warf[k].append(i)

            #calculate reward for bandit
            reg = 1
            if l[i] > 7.0 and bandit_action == 2:
                reg = 0
            elif abs(l[i]- 5.0) <= 2.0 and bandit_action == 1:
                reg = 0
            elif l[i] < 3.0 and bandit_action == 0:
                reg = 0
            regret[2].append(reg)
            num_correct[2] += (1-reg)

            #update bandit params, update frac correct
            treat_bandit.update_params(patient, bandit_action, 10*(1-reg))
            for q in range(3):
                frac_correct[q].append(float(num_correct[q])/(i+1))

        for q in range(3):
            regret_list = regret[q]
            frac_list = frac_correct[q]
            regret_sum = [sum(regret_list[0:i+1]) for i in range(len(regret_list))]
            regret_per_run[q][k] = regret_sum
            frac_per_run[q][k] = frac_list


    with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
        for i in range(3):
            r = pd.DataFrame(np.array(regret_per_run[i]))
            r.to_excel(writer, sheet_name='regret_'+str(i))
            b = pd.DataFrame(np.array(frac_per_run[i]))
            b.to_excel(writer, sheet_name='frac_'+str(i))
            
        a = np.array(regret_per_run[1])             
        b = np.array(regret_per_run[2])
        b = np.hstack((np.zeros((NUM_RUNS,1)),b))
        b = [[b[i,j+1]-b[i,j] for j in range(len(p[:,0]))] for i in range(NUM_RUNS)]
        b = np.array(b)

        aa = np.array(regret_per_run[0])
        aa = np.hstack((np.zeros((NUM_RUNS,1)),aa))
        aa = [[aa[i,j+1]-aa[i,j] for j in range(len(p[:,0]))] for i in range(NUM_RUNS)]
        aa = np.array(aa)
        
        c = []
        d = []
        for j in range(NUM_RUNS):
            c.append(b[j, is_warf[j]])
            d.append(aa[j, is_warf[j]])

        for j in range(NUM_RUNS):
            x = c[j]
            x = [sum(x[0:i+1]) for i in range(len(x))]
            c[j] = x
            x = d[j]
            x = [sum(x[0:i+1]) for i in range(len(x))]
            d[j] = x
            
        xx = []
        for j in range(NUM_RUNS): #order goes: warfarin, warfarin..., us, us, us...
            xx.append(a[j,:])
        for j in range(NUM_RUNS):
            xx.append(c[j])
        for j in range(NUM_RUNS):
            xx.append(d[j])
        r = pd.DataFrame(np.array(xx))
        r.to_excel(writer, sheet_name='regret_warfarin_cmp')

