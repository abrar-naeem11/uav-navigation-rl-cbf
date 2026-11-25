import pandas as pd
import matplotlib.pyplot as plt

# Load monitor.csv
df = pd.read_csv("monitor.csv", skiprows=1)  # first line is metadata

# The dataframe has columns: r (episode reward), l (episode length), t (time steps)
print(df.head())

# Plot episode rewards
plt.figure()
plt.plot(df["r"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward")
plt.show()

# Plot episode lengths
plt.figure()
plt.plot(df["l"])
plt.xlabel("Episode")
plt.ylabel("Length")
plt.title("Episode Length")
plt.show()