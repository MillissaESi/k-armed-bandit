import numpy as np
import matplotlib.pyplot as plt
from math import exp


class armed_bandit:
    def __init__(self):
        # The average of gain for every machine
        self.esp_values = np.random.normal(0, 1, 10)
        # store the number of occurences a machine was played before
        self.k = np.zeros(10)
        # to store the average of rewards
        self.avrg_rwd = np.zeros(10)
        # an array to store the preference functions
        self.h = [0 for i in range(10)]
        # pr ^references
        self.porb_preference = np.zeros(10)

    def get_rwd(self, action):
        return np.random.normal(self.esp_values[action], 1)

    def choose_action(self, e):
        # random float btw 0-1
        rnd = np.random.random()
        if (rnd < e):
            action = np.random.randint(10)
        else:
            # pick the machine with the greatest reward
            action = np.argmax(self.avrg_rwd)
        return action

    def update_avrg_rwd(self, action, rwd):
        self.k[action] += 1
        self.avrg_rwd[action] += (1 / self.k[action]) * (rwd - self.avrg_rwd[action])

    def update_porb_preference(self):
        for i in range(10):
            self.porb_preference[i] = exp(self.h[i]) / sum(exp(self.h[j]) for j in range(10))

    def update_h(self, action, rwd):
        for i in range(10):
            if (i == action):
                self.h[i] = self.h[i] + 0.1 * (rwd - self.avrg_rwd[action]) * (1 - self.porb_preference[i])
            else:
                self.h[i] = self.h[i] - 0.1 * (rwd - self.avrg_rwd[i]) * self.porb_preference[i]

    def choose_action_prefered(self):
        rand_action = np.random.rand()
        continu = True
        born_inf = 0
        i = 0
        while (continu):
            born_sup = born_inf + self.porb_preference[i]
            if (rand_action >= born_inf and rand_action <= born_sup):
                action = i
                continu = False
            else:
                born_inf = born_sup
                i = i + 1
        return action


def run_prefered(bandits, nb_iterations, e):
    iterations = []
    optimal = []
    for i in range(nb_iterations):
        bandits.update_porb_preference()
        action = bandits.choose_action_prefered()
        rwd = bandits.get_rwd(action)
        bandits.update_avrg_rwd(action, rwd)
        bandits.update_h(action, rwd)
        optimal_action = np.argmax(bandits.esp_values)
        if (action == optimal_action):
            optimal.append(1)
        else:
            optimal.append(0)
        iterations.append(rwd)

    return np.array(optimal)


def run(bandits, nb_iterations, e):
    iterations = []
    optimal = []
    for i in range(nb_iterations):
        action = bandits.choose_action(e)
        rwd = bandits.get_rwd(action)
        bandits.update_avrg_rwd(action, rwd)
        optimal_action = np.argmax(bandits.esp_values)
        if (action == optimal_action):
            optimal.append(1)
        else:
            optimal.append(0)
        iterations.append(rwd)
    return np.array(optimal)


"""
 Prepare the environment 
"""
nb_iterations = 1000
nb_runs = 500
e1 = 0.1
e2 = 0
avrg_eps_value1 = np.zeros(nb_iterations)
avrg_eps_value2 = np.zeros(nb_iterations)

for i in range(nb_runs):
    bandits1 = armed_bandit()
    avrg_eps_value1 += run(bandits1, nb_iterations, e1)
    bandits2 = armed_bandit()
    avrg_eps_value2 += run_prefered(bandits2, nb_iterations, e2)

avrg_eps_value1 /= np.float(nb_runs)
avrg_eps_value2 /= np.float(nb_runs)

"""
average rewards 
"""
plt.plot(avrg_eps_value1, label="eps = 0.0")
plt.plot(avrg_eps_value2, label="bandit gradient")
plt.ylim(0, 1)
plt.legend()
plt.show()
