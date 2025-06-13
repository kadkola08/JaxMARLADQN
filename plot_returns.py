import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

map_name = "HEURISTICENEMYSMAX_2S3Z"

win_rates = np.zeros((9, 500))

for i in range(9):
    df = pd.read_csv(f"exps/{map_name}_i_{i}.csv")
    win_rates[i] = df["test_returned_won_episode"].to_list()

mean = np.mean(win_rates, axis=0)
std = np.std(win_rates, axis=0) 
n = len(win_rates)

ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
lower_ci = ci[0]
upper_ci = ci[1]

x = np.arange(500) / 500  # X-axis (0 to 499)
plt.plot(x, mean, 'b-', linewidth=2, label='Modified QMIX')
plt.fill_between(x, lower_ci, upper_ci, alpha=0.3, color='blue', label='95% CI')
plt.savefig(f"exps/{map_name}_win_rates.png")
plt.show()

