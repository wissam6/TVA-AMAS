import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("results3.csv")

# Extract columns
rounds = df["num_voters"]
risk = df["strategic_voting_risk"]
happiness = df["overall_happiness"]

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(rounds, risk, marker='o', label="Strategic Voting Risk")
plt.plot(rounds, happiness, marker='o', label="Overall Happiness")
plt.xlabel("Number of Voters")
plt.ylabel("Value")
plt.title("Strategic Voting Risk and Overall Happiness vs. Number of Voters (Max Rounds = 15, Alternatives = 15)")
plt.legend()
plt.grid(True)
plt.show()
