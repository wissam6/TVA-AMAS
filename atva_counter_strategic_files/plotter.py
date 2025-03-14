import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("results.csv")

# Extract columns
rounds = df["max_rounds"]
risk = df["strategic_voting_risk"]
happiness = df["overall_happiness"]

# Create the plot
plt.figure(figsize=(10,6))
plt.plot(rounds, risk, marker='o', label="Strategic Voting Risk")
plt.plot(rounds, happiness, marker='o', label="Overall Happiness")
plt.xlabel("Number of Voting Rounds")
plt.ylabel("Value")
plt.title("Strategic Voting Risk and Overall Happiness vs. Voting Rounds (20 Voters, 50 Alternatives)")
plt.legend()
plt.grid(True)
plt.show()
