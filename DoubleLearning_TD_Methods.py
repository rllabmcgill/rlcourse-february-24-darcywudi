
# coding: utf-8

# In this code,Expected Sarsa, Two Step Sarsa, double sarsa, and double expected sarsa are implemented
# I use the previous implementation of q, sarsa and double q (By weiwei Zhang) as comparison baslines 
# I use the same problem setting as the baselines, and this implementation uses some of the code from weiwei's implemtion on q,double q, and sarsa
# In[80]:



import numpy as np
from Baseline import CliffWalking
from Baseline import qLearning
from Baseline import dqLearning
from Baseline import Sarsa

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
gamma = 0.8


# ## Expected Sarsa

# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def ExpSarsa(cw, width, height, avgR, iterator, max_iter):
    q = np.zeros((width, height, 4))
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = q[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
        a1 = q[s1[0], s1[1]].argmax()
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
        q[s0[0], s0[1], a] +=  0.1 * (r + gamma * 0.25*(q[s1[0], s1[1], 0] + q[s1[0], s1[1], 1] + q[s1[0], s1[1], 2] + q[s1[0], s1[1], 3] ) - q[s0[0], s0[1], a])
    iterator.append(i)


# ## Double Sarsa

# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def dSarsa(cw, width, height, avgR, iterator, max_iter):
#    q = np.zeros((width, height, 4))
    qA = np.zeros((width, height, 4))
    qB = np.zeros((width, height, 4))
    qC = np.zeros((width, height, 4))
    
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = qA[s0[0], s0[1]].argmax()
        if np.random.random() < 0.5:
            a = qB[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
         
        
        qC = 0.5*(qA + qB)
        a1 = qC[s1[0], s1[1]].argmax()
        
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
        qA[s0[0], s0[1], a] +=  0.1 * (r + gamma * qB[s1[0], s1[1], a1] - qA[s0[0], s0[1], a])
        
        if np.random.random() <= 0.5:
                temp = qA
                qA = qB
                qB = temp
    iterator.append(i)


# ## Sarsa

# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def dExpSarsa(cw, width, height, avgR, iterator, max_iter):
    qA = np.zeros((width, height, 4))
    qB = np.zeros((width, height, 4))
    qC = np.zeros((width, height, 4))
    
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a = qA[s0[0], s0[1]].argmax()
        if np.random.random() < 0.5:
            a = qB[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a = np.random.choice(range(4))
        r = cw.move(actions[a])
        G += r
        i += 1
        if i <= max_iter:
            avgR[i] = G / i
        s1 = cw.getPosition()
         
        
        qC = 0.5*(qA + qB)
        a1 = qC[s1[0], s1[1]].argmax()
        
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
            qA[s0[0], s0[1], a] +=  0.1 * (r + gamma * 0.25*(qB[s1[0], s1[1], 0] + qB[s1[0], s1[1], 1] + qB[s1[0], s1[1], 2] + qB[s1[0], s1[1], 3] ) - qA[s0[0], s0[1], a])
        if np.random.random() <= 0.5:
                temp = qA
                qA = qB
                qB = temp
    iterator.append(i)



# ## Multi Step Sarsa

# In[83]:

actions = ['u', 'd', 'r', 'l']
epsilon = 0.1
#Sarsa
def TwoSarsa(cw, width, height, avgR, iterator, max_iter):
    q = np.zeros((width, height, 4))
    G = 0.0
    i = 0
    while not cw.isEnd():
        s0 = cw.getPosition()
        a0 = q[s0[0], s0[1]].argmax()
        if np.random.random() < epsilon:
            a0 = np.random.choice(range(4))
        r0 = cw.move(actions[a0])
        G += r0
        i += 1
        
        s1 = cw.getPosition()
        a1 = q[s1[0], s1[1]].argmax()
        if np.random.random() < epsilon:
            a1 = np.random.choice(range(4))
        r1 = cw.move(actions[a1])

        s2 = cw.getPosition()
        a2 = q[s2[0], s2[1]].argmax()
        if np.random.random() < epsilon:
            a2 = np.random.choice(range(4))
        G += r1 - gamma*gamma*q[s2[0], s2[1], a2]
        if i < max_iter:
            avgR[i] = G / i
        q[s0[0], s0[1], a0] +=  0.1 * (r0 +   gamma*gamma * q[s2[0], s2[1], a2] - q[s0[0], s0[1], a0])
    iterator.append(i)




# ## Sarsa, Q-learning and Double Q-learning on CliffWalking

# In[90]:

import matplotlib.pyplot as plt
#initialize CliffWalking
cw = CliffWalking(variance = 1)
ite1 = []
avgR1 = [[0.0]*500] * 500
ite2 = []
avgR2 = [[0.0]*500] * 500
for i in range(500):
    cw.resetPosition()
    qLearning(cw, 12, 4, avgR1[i], ite1, 499)
    cw.resetPosition()
    dqLearning(cw, 12, 4, avgR2[i], ite2, 499)
plt.plot(np.mean(np.asarray(avgR1, dtype=np.float32), axis=0), label = 'QLearning')
print "Average iteration of Q-learning: " + str(np.mean(ite1))
plt.plot(np.mean(np.asarray(avgR2, dtype=np.float32), axis=0), label = 'Double QLearning')
print "Average iteration of Double Q-learning: " + str(np.mean(ite2))
plt.ylabel('Reward per episode')
plt.xlabel('Episodes')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ## Sarsa, Q-learning and Double Q-learning on randomized CliffWalking

# In[92]:

#initialize Random CliffWalking
variances = [0, 0.1, 0.5, 1, 5, 10]
ite1_a = []
ite2_a = []
ite3_a = []
ite4_a = []
ite5_a = []
ite6_a = []
ite7_a = []

for variance in variances:
    cw = CliffWalking(variance = variance)
    ite1 = []
    avgR1 = [[0.0]*200] * 100
    ite2 = []
    avgR2 = [[0.0]*200] * 100
    ite3 = []
    avgR3 = [[0.0]*200] * 100
    ite4 = []
    avgR4 = [[0.0]*200] * 100
    ite5 = []
    avgR5 = [[0.0]*200] * 100
    ite6 = []
    avgR6 = [[0.0]*200] * 100
    ite7 = []
    avgR7 = [[0.0]*200] * 100
    
	
    for i in range(100):
        cw.resetPosition()
        qLearning(cw, 12, 4, avgR1[i], ite1, 199)
        cw.resetPosition()
        dqLearning(cw, 12, 4, avgR2[i], ite2, 199)
        cw.resetPosition()
        Sarsa(cw, 12, 4, avgR3[i], ite3, 199)
        cw.resetPosition()
        ExpSarsa(cw, 12, 4, avgR4[i], ite4, 199)
        cw.resetPosition()
        TwoSarsa(cw, 12, 4, avgR5[i], ite5, 199)
        cw.resetPosition()
        dSarsa(cw, 12, 4, avgR6[i], ite6, 199)
        cw.resetPosition()
        dExpSarsa(cw, 12, 4, avgR6[i], ite6, 199)
		
    ite1_a.append(np.mean(ite1))
    ite2_a.append(np.mean(ite2))
    ite3_a.append(np.mean(ite3))
    ite4_a.append(np.mean(ite4))
    ite5_a.append(np.mean(ite5))
    ite6_a.append(np.mean(ite6))
    ite7_a.append(np.mean(ite7))
	
	
    print "Variance:" + str(variance)
    print "Average iteration of Q-learning: " + str(np.mean(ite1))
    print "Average iteration of Double Q-learning: " + str(np.mean(ite2))
    print "Average iteration of Sarsa: " + str(np.mean(ite3))
    print "Average iteration of Expected Sarsa: " + str(np.mean(ite4))
    print "Average iteration of Two Step Sarsa: " + str(np.mean(ite5))
    print "Average iteration of Double Sarsa: " + str(np.mean(ite6))
    print "Average iteration of Double Expected Sarsa: " + str(np.mean(ite7))
    
	


plt.plot(ite1_a, label = 'QLearning')
plt.plot(ite2_a, label = 'Double QLearning')
plt.plot(ite3_a, label = 'Sarsa')
plt.plot(ite4_a, label = 'Expected Sarsa')
plt.plot(ite5_a, label = 'Two-Step Sarsa')
plt.plot(ite6_a, label = 'Double Sarsa')
plt.plot(ite6_a, label = 'Double Expected Sarsa')



plt.xticks(range(6), variances, rotation=0)  
plt.ylabel('Number of steps')
plt.xlabel('Variances')
plt.legend(loc='best')
plt.tight_layout()
plt.show()



# Plot the average return
plt.plot(ite1_a, label = 'QLearning')
plt.plot(ite3_a, label = 'Sarsa')
plt.plot(ite4_a, label = 'Expected Sarsa')
plt.plot(ite5_a, label = 'Two-Step Sarsa')

plt.xticks(range(6), variances, rotation=0)  
plt.ylabel('Number of steps')
plt.xlabel('Variances')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ## Discussion
# We create a randomized CliffWalking environment and compare these methods based on different vairance ratio. From the experiments,Double Q-learning is less sensitive to the variances (the higer variance the more uncertain environment) and needs much fewer steps to reach the goal when the variance is big.

# In[ ]:



