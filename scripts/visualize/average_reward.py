import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("monitor.csv", skiprows=1)  # first row is metadata
rewards = df["r"].to_numpy()   # episode rewards

def moving_average(x, window_size=50):
    return np.convolve(x, np.ones(window_size)/window_size, mode="valid")

avg_rewards = moving_average(rewards, window_size=50)
plt.figure(figsize=(10,6))
plt.plot(avg_rewards, label="Average Reward (window=50)")
plt.plot(rewards, alpha=0.3, label="Raw Reward", color="gray")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Average Reward Curve")
plt.legend()
plt.show()