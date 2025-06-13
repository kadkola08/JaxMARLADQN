import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_curves(arr_list, legend_list, color_list, ylabel, fig_title):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        ylabel (string): label of the Y axis
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Steps (10 mil)")

    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err *= 1.96
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3,
                        color=color)
        # save the plot handle
        h_list.append(h)

    # plot legends
    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)
    
    return fig, ax

map_name = "HEURISTICENEMYSMAX_SMACV2_10_UNITS_FIX_POS"
# alg_name = "mod_qmix"

win_rates_van = np.zeros((10, 500))
win_rates_mod = np.zeros((10, 500))

for i in range(10):
    df = pd.read_csv(f"exps/mod_qmix_{map_name}_i_{i}.csv")
    win_rates_mod[i] = df["test_returned_won_episode"].to_list()
print("mod")

for i in range(10):
    df = pd.read_csv(f"exps/van_qmix_{map_name}_i_{i}.csv")
    win_rates_van[i] = df["test_returned_won_episode"].to_list()
print("van")

x_ticks = np.arange(0, 500, 100)
x_labels = [f"{x/500:.1f}" for x in x_ticks]

fig, ax = plot_curves(
    [win_rates_mod, win_rates_van],                  # Single array in a list
    ["Modified QMIX", "Vanilla QMIX"],            # Single legend entry
    ["blue", "red"],                     # Single color
    "Test Win Rate",              # Y-axis label
    f"Win Rate on {map_name}"     # Figure title
)

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)

plt.savefig(f"exps/{map_name}_win_rates.png")
plt.show()
