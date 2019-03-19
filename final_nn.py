#REMEMBER TO RUN ME IN PYTHON 3!!!

import numpy as np
import pandas as pd
import random
import tensorflow as tf
import threading

class Neural_Bandit:
    def __init__(self, patient_params):
        #note a shape of 1 because we're online
        self.x = tf.placeholder(shape=[1,patient_params], dtype=tf.float32)
        self.y = tf.placeholder(shape=[1], dtype=tf.int64)
        #self.y = tf.placeholder(shape=[1,3], dtype=tf.int64)

        #define network
        out = self.x
##        out = tf.layers.dense(out, units=10000, activation=tf.nn.relu)
##        out = tf.layers.dense(out, units=1000, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=1000, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=100, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=10, activation=None)
        out = tf.layers.dense(out, units=3, activation=None)

        #accuracy/loss
        self.preds = tf.argmax(out, axis=1)

        #may want to change loss to binary correct/incorrect
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.y, out)



net_regret = [[] for i in range(5)]

def run_NN(l_r, shuffle_list, idx1, idx2):
    num_correct = 0
    frac_correct = []
    regret = []

    model = Neural_Bandit(len(p[0,:]))
    optimizer = tf.train.AdamOptimizer(l_r)
    train_op = optimizer.minimize(model.loss)
    init_op = tf.global_variables_initializer()

    random.shuffle(shuffle_list)
    patients, labels = zip(*shuffle_list)
    patients = np.array(patients)
    labels = np.concatenate(labels)

    with tf.Session() as sess:
        sess.run(init_op)

        # Main loop
        for i in range(len(patients[:,0])):
            patient = patients[i,:].reshape((1,-1))
            label = labels[i]
            label_NN = np.zeros((1,3), dtype=int)
            label_NN[0,label] = 1
            
            _, loss, pred = sess.run(
                [train_op, model.loss, model.preds], feed_dict={model.x: patient, model.y: np.array([label])})
            
            dose = pred[0]
            
            reg = 1
            if label == dose:
                reg = 0
                num_correct += 1
                
            regret.append(reg)
            frac_correct.append(float(num_correct)/(i+1))

    net_regret[idx1].append(regret)



if __name__ == '__main__':
    #hyperparams
    learning_rate = [0.08, 0.04, 0.01, 0.001, 0.0001]

    NUM_TRIALS = 10
    
    #import data
    print("Importing Data")
    f = 'warfarin_binary_supersparse.xlsx' 
    xl = pd.ExcelFile(f)
    df1 = xl.parse('Sheet1') #change to Sheet1!
    p = df1.values

    f = 'warfarin_orig.xlsx'
    xl = pd.ExcelFile(f)
    df1 = xl.parse('Sheet2')
    l = df1.values
    l[l < 3.0] = 0
    l[abs(l-5.0) <= 2.0] = 1
    l[l > 7.0] = 2
    assert(np.amax(l) == 2)
    l = l.astype(int)

    print("Starting runs")
    shuffle_list = list(zip(p, l))

    t_set = [[threading.Thread(target=run_NN, args=(learning_rate[i], shuffle_list, i,j)) for i in range(5)] for j in range(NUM_TRIALS)]
    for j in range(10):
        thr = t_set[j]
        for i in range(5):
            thr[i].start()
        for i in range(5):
            thr[i].join()
        print("Set: " +str(j+1)+" out of 10 done!")

    with pd.ExcelWriter('output_NN_arc2.xlsx', engine='openpyxl') as writer:
        for i in range(5):
            r = pd.DataFrame(np.array(net_regret[i]))
            r.to_excel(writer, sheet_name='regret_'+str(learning_rate[i]))            
























            
