import numpy as np
import matplotlib.pyplot as plt


class armed_bandit:
    def __init__(self):
        self.esp_values = np.random.normal(0, 1, 10)
        self.k = np.zeros(10)
        self.avrg_rwd = np.zeros(10)

    def get_rwd(self, action):
        return np.random.normal(self.esp_values[action], 1)

    def choose_action(self, e):
        rnd = np.random.random()
        if (rnd < e):
            action = np.random.randint(10)
        else:
            action = np.argmax(self.avrg_rwd)
        return action

    def update_avrg_rwd(self, action, rwd):
        self.k[action] += 1
        self.avrg_rwd[action] += (1 / self.k[action]) * (rwd - self.avrg_rwd[action])


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
    return np.array(optimal), np.array(iterations)

"""
 prepare environment 
"""

nb_iterations = 2000
nb_runs = 1000
e1 = 0
e2 = 0.1
e3 = 0.01

avrg_eps_reward1 = np.zeros(nb_iterations)
avrg_eps_reward2 = np.zeros(nb_iterations)
avrg_eps_reward3 = np.zeros(nb_iterations)

avrg_eps_optimal1 = np.zeros(nb_iterations)
avrg_eps_optimal2 = np.zeros(nb_iterations)
avrg_eps_optimal3 = np.zeros(nb_iterations)

for i in range(nb_runs):
    bandits1 = armed_bandit()
    optimal, iterations = run(bandits1, nb_iterations, e1)
    avrg_eps_reward1 += iterations
    avrg_eps_optimal1 += optimal

    bandits2 = armed_bandit()
    optimal, iterations = run(bandits2, nb_iterations, e2)
    avrg_eps_reward2 += iterations
    avrg_eps_optimal2 += optimal

    bandits3 = armed_bandit()
    optimal, iterations = run(bandits3, nb_iterations, e3)
    avrg_eps_reward3 += iterations
    avrg_eps_optimal3 += optimal

avrg_eps_reward1 /= np.float(nb_runs)
avrg_eps_optimal1 /= np.float(nb_runs)

avrg_eps_reward2 /= np.float(nb_runs)
avrg_eps_optimal2 /= np.float(nb_runs)

avrg_eps_reward3 /= np.float(nb_runs)
avrg_eps_optimal3 /= np.float(nb_runs)

"""
  The average rewards 
"""

plt.plot(avrg_eps_reward1, label="eps = 0.0")
plt.plot(avrg_eps_reward2, label="eps = 0.1")
plt.plot(avrg_eps_reward3, label="eps = 0.01")

plt.ylim(0, 1.6)
plt.legend()
plt.show()

"""
  The frequency of selecting better machines 
"""

plt.plot(avrg_eps_optimal1, label="eps = 0.0")
plt.plot(avrg_eps_optimal2, label="eps = 0.1")
plt.plot(avrg_eps_optimal3, label="eps = 0.01")

plt.ylim(0, 1)
plt.legend()
plt.show()

