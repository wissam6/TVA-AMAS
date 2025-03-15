import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from btva import BTVA  
from happiness import binary_happiness 

def generate_random_preferences_matrix(num_alternatives, num_voters):
    return np.array([np.random.permutation(num_alternatives) for _ in range(num_voters)]).T

    
def run_non_strategic_election(self):
    scores = np.zeros(self.num_alternatives)
    for voter in range(self.num_voters):
        top_choice = self.preference_matrix[0, voter]  
        scores[top_choice] += 1
    
    election_ranking = np.argsort(-scores, kind='stable')
    return election_ranking

def run_strategic_election(self, election_result):
    winner = election_result[0]
    tactical_voting_risk = np.zeros(self.num_voters)

    scores = np.zeros(self.num_alternatives)
    for voter in range(self.num_voters):
        top_choice = self.preference_matrix[0, voter]
        scores[top_choice] += 1

    for voter in range(self.num_voters):
        voter_preference = self.preference_matrix[:, voter]
        favorite_candidate = voter_preference[0]

        if favorite_candidate != winner:
            stronger_opponent = None
            for candidate in voter_preference[1:]:
                if scores[candidate] > scores[favorite_candidate]:
                    stronger_opponent = candidate
                    break
            if stronger_opponent is not None:
                tactical_voting_risk[voter] = 1

    return np.sum(tactical_voting_risk) / self.num_voters


num_voters = 20
candidate_range = range(2, 51)  
results = []

# Loop over all candidates in candidate_range and calculate the overall happiness and risk fo strategic voting
for num_candidates in candidate_range:
    preference_matrix = generate_random_preferences_matrix(num_candidates, num_voters)  

    btva = BTVA(preference_matrix, binary_happiness)
    election_result = run_non_strategic_election(btva)
    happinesses = btva.calc_happinesses(election_result)
    risk_of_strategic_voting = run_strategic_election(btva,election_result)
    results.append([num_candidates, np.mean(happinesses), risk_of_strategic_voting])

df_results = pd.DataFrame(results, columns=["Num Candidates", "Avg Happiness", "Risk of Strategic Voting"])

plt.figure(figsize=(10, 5))
plt.plot(df_results["Num Candidates"], df_results["Avg Happiness"], label="Average Happiness", marker='o')
plt.plot(df_results["Num Candidates"], df_results["Risk of Strategic Voting"], label="Risk of Strategic Voting", marker='s', linestyle='dashed', color='orange')

plt.xlabel("Number of Candidates")
plt.ylabel("Value")
plt.title("Risk of Strategic Voting vs Voter Happiness")
plt.legend()
plt.grid(True)
plt.show()