# k-armed-bandit
### 1. Epsilon Greedy: 

![greedy](/images/fig8.png)

The average rewards for 2000 runs 

![greedy](/images/fig5.png)

Frequency of selecting the best machines 

For eps = 0.1,  the agent reaches his maximum average reward more quickly because the agent explores new actions 10% of his time and exploits his knowledge of the environment 90% of the remaining time, so he is more likely to fall on machines that are more efficient.

For eps = 0.01, the agent explores his environment less (only 1% of the time and uses his knowledge 99% of the remaining time). It is therefore less likely to come across new, more efficient machines.

For eps = 0 the agent never explores his environment, he only uses his knowledge of his environment. This means that his average reward remains constant and does not improve. Here are the results for playing 2000 times: 

![greedy](/images/fig4.png)


For eps = 0.01, after about 1100 coins played, the agents starts to gain a higher average of rewards compared to eps = 0.1. This comes from the fact that the agent, after having explored his environment so much, needs to focus more on exploiting the knowledge he earned about his environment and improve his strategy in selecting better machines (The agent can no longer enrich his knowledge through exploration). In order to optimize the behavior of the agent, he would have to learn faster at the beginning and exploit his knowledge thereafter, for that it's better to define a decreasing function for eps.

### 2. Optimistic initialization:

![optimistic](/images/fig7.png)

The average rewards for 2000 runs 

![optimistic](/images/fig6.png)

Frequency of selecting the best machines 

For (Q1 = 5 and eps = 0), the agent chooses the best machine more frequently. In fact, an optimistic initialization encourage the agent to explore the machines that seem promising. In case these machines are not really promising, the agent lowers his estimate for these machines. We force the agent to explore each machine a certain number of times which will lead to convergence.

### 3. Confidence Interval:

![interval](/images/fig1.png)

The average rewards for 2000 runs

![interval](/images/fig2.png)

Frequency of selecting the best machines 


For (eps = 0 and c = 2) the agent chooses the best machine more frequently. Indeed, the agent will no longer have an interest in revisiting the machines that are bad by testing them. This forces him to explore new machines that appear to him more promising.

### 4. Gradient climb:

![interval](/images/fig3.png)

Using a preference function, the agent chooses the best machine more frequently. At the beginning the agent has an equal preference for all machines, so each machine has the same probability of being chosen. If the agent falls on a machine that is bad, his preference for this machine decreases and therefore the probability of choosing it also decreases.


