import numpy as np
import matplotlib.pyplot as plt
from math import log , sqrt


class armed_bandit:
    def __init__(self, optimistic, c):
        self.esp_values = np.random.normal(0, 1, 10)
        self.k = np.zeros(10)
        self.avrg_rwd = [optimistic for _ in range(10)]
        self.avrg_rwd_confidence = np.zeros(10)
        self.c = c

    def get_rwd(self, action):
        return np.random.normal(self.esp_values[action], 1)

    def choose_action(self, e, t):
        rnd = np.random.random()
        if (rnd < e):
            action = np.random.randint(10)
        else:
            # calculate the confidence intervals
            for i in range(len(self.avrg_rwd)):
                self.avrg_rwd_confidence[i] = self.avrg_rwd[i] + self.c * sqrt(log(t, 10) / self.k[i])
                # pick the machine with the highest reward
            action = np.argmax(self.avrg_rwd_confidence)
        return action

    def update_avrg_rwd(self, action, rwd):
        self.k[action] += 1
        self.avrg_rwd[action] += 0.1 * (rwd - self.avrg_rwd[action])

def run( bandits , nb_iterations , e ):
    iterations =[]
    optimal=[]
    t=0
    for i in range(nb_iterations) :
        t=t+1
        action = bandits.choose_action(e,t)
        rwd = bandits.get_rwd(action)
        bandits.update_avrg_rwd(action,rwd)
        optimal_action = np.argmax(bandits.esp_values)
        if (action==optimal_action):
            optimal.append(1)
        else:
             optimal.append(0)
        iterations.append(rwd)
    return np.array(optimal) , np.array(iterations)

"""
prepare the environment 
"""
nb_iterations = 1000
nb_runs = 500
e1=0
e2=0.1
avrg_eps_reward1 = np.zeros(nb_iterations)
avrg_eps_reward2 = np.zeros(nb_iterations)
avrg_eps_optimal1 = np.zeros(nb_iterations)
avrg_eps_optimal2 = np.zeros(nb_iterations)

for i in range(nb_runs):
    bandits1 = armed_bandit(0, 0.5)
    optimal, iterations = run(bandits1, nb_iterations, e1)
    avrg_eps_reward1 += iterations
    avrg_eps_optimal1 += optimal
    bandits2 = armed_bandit(0, 0)
    optimal, iterations = run(bandits2, nb_iterations, e2)
    avrg_eps_reward2 += iterations
    avrg_eps_optimal2 += optimal

avrg_eps_reward1 /= np.float(nb_runs)
avrg_eps_optimal1 /= np.float(nb_runs)
avrg_eps_reward2 /= np.float(nb_runs)
avrg_eps_optimal2 /= np.float(nb_runs)


"""
average rewards 
"""
plt.plot(avrg_eps_reward1,label="eps = 0.0 c=2")
plt.plot(avrg_eps_reward2,label="eps = 0.1")
plt.ylim(0,2)
plt.legend()
plt.show()

"""
The frequence of choosing the best machine 
"""
plt.plot(avrg_eps_optimal1,label="eps = 0.0 c=2")
plt.plot(avrg_eps_optimal2,label="eps = 0.1")
plt.ylim(0,1)
plt.legend()
plt.show()