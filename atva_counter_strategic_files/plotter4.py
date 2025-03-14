import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("results4.csv")

# Extract columns
rounds = df["num_alternatives"]
risk = df["strategic_voting_risk"]
happiness = df["overall_happiness"]

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(rounds, risk, marker='o', label="Strategic Voting Risk")
plt.plot(rounds, happiness, marker='o', label="Overall Happiness")
plt.xlabel("Number of Alternatives")
plt.ylabel("Value")
plt.title("Strategic Voting Risk and Overall Happiness vs. Number of Alternatives (Max Rounds = 15, Voters = 15)")
plt.legend()
plt.grid(True)
plt.show()
