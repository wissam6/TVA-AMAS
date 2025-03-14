import numpy as np
import itertools
from happiness import *
from btva import BTVA
import matplotlib.pyplot as plt

###############################################################################
# 1. Advanced Voting System: Addressing Voter Collusion (Multiple-Voter Manipulations)
###############################################################################

# make it work for diff group sizes and maybe dont do it in case the voter is already satisfied.
# it currently uses only compromising/burying
class BTVA_Collusion(BTVA):
    def run_collusive_strategic_voting(self, group_size):  
        """
        Check if groups of voters (of size group_size) can coordinate
        their tactical vote to change the election outcome.
        """
        # Run the baseline election with voters' true (non-strategic) preferences
        original_result = self.run_non_strategic_election()
        # Identify the original winner (the candidate ranked first in the results)
        original_winner = original_result[0, 0]
        # Initialize an array to track which voters would benefit from collusion
        collusion_incentives = np.zeros(self.num_voters)
        
        # Generate all combinations of voters of the specified group size
        for group in itertools.combinations(range(self.num_voters), group_size):
            # Create a copy of the original preference matrix to modify ballots for this group
            new_pref_matrix = np.copy(self.preference_matrix)
            # Dictionary to hold cumulative scores for each candidate (excluding the original winner)
            candidate_scores = {}
            # Evaluate each voter in the group
            for voter in group:
                # For each candidate in the voter's ranking
                for candidate in self.preference_matrix[:, voter]:
                    if candidate != original_winner:
                        # Find the candidate's rank (position) in the voter's ballot
                        rank = np.where(self.preference_matrix[:, voter] == candidate)[0][0]
                        # Sum the rank scores over the group; lower score indicates a more preferred candidate
                        candidate_scores[candidate] = candidate_scores.get(candidate, 0) + rank
            # If no candidate was found (unlikely), skip this group
            if not candidate_scores:
                continue
            # Choose the candidate with the lowest total rank as the collusive candidate
            collusive_candidate = min(candidate_scores, key=candidate_scores.get)
            
            # For each voter in the colluding group, modify their ballot:
            # Place the collusive candidate first and the original winner last.
            for voter in group:
                voter_pref = self.preference_matrix[:, voter]
                # Build a new ballot:
                # 1. The collusive candidate at the top
                # 2. Then all other candidates (preserving their order) except the collusive candidate and original winner
                # 3. Finally, the original winner at the bottom
                new_ballot = [collusive_candidate] + [c for c in voter_pref if c not in [collusive_candidate, original_winner]] + [original_winner]
                # Update the modified preference matrix with the new ballot for this voter
                new_pref_matrix[:, voter] = new_ballot
            
            # Create a new election instance with the modified ballots
            collusive_btva = BTVA(self.voting_scheme, new_pref_matrix)
            # Run the election with the collusive ballots
            new_result = collusive_btva.run_non_strategic_election()
            # Identify the new winner after the collusive move
            new_winner = new_result[0, 0]
            # If the new winner is different, the collusion is successful
            if new_winner != original_winner:
                # Mark each voter in the group as having an incentive for collusion
                for voter in group:
                    collusion_incentives[voter] = 1
                print(f"Collusion by voters {group} can change the winner from {original_winner} to {new_winner}")
        # Return the incentives array indicating which voters benefit from collusion
        return collusion_incentives


###############################################################################
# Example Usage
###############################################################################
pref_matrix = np.array([
    [0, 1, 3, 0, 2, 3, 2, 3],
    [2, 0, 0, 1, 4, 4, 4, 2],
    [1, 3, 2, 2, 1, 0, 0, 1],
    [3, 4, 4, 3, 3, 1, 1, 4],
    [4, 2, 1, 4, 0, 2, 3, 0]
])

def generate_random_preference_matrix(num_voters, num_candidates):
    preference_matrix = np.zeros((num_candidates, num_voters), dtype=int)
    for voter in range(num_voters):
        preference_matrix[:, voter] = np.random.permutation(num_candidates)
    return preference_matrix

def run_experiments(voting_scheme, num_voters, num_candidates, group_size, num_trials):
    collusion_success_count = 0
    total_collusion_incentives = np.zeros(num_voters)
    total_happinesses = np.zeros(num_voters)
    happiness_changes = []

    for _ in range(num_trials):
        preference_matrix = generate_random_preference_matrix(num_voters, num_candidates)
        btva_collusion = BTVA_Collusion(voting_scheme, preference_matrix)
        collusion_incentives = btva_collusion.run_collusive_strategic_voting(group_size)
        
        if np.sum(collusion_incentives) > 0:
            collusion_success_count += 1
        
        # Calculate happiness after collusion
        election_result = btva_collusion.run_non_strategic_election()
        election_ranking, _ = election_result
        happinesses = btva_collusion.calc_happinesses(preference_matrix, election_ranking, exponential_decay_happiness)
        
        total_collusion_incentives += collusion_incentives
        total_happinesses += happinesses
        happiness_changes.append(happinesses)
    
    average_collusion_incentives = total_collusion_incentives / num_trials
    collusion_success_rate = collusion_success_count / num_trials
    average_happinesses = total_happinesses / num_trials
    
    return collusion_success_rate, average_collusion_incentives, average_happinesses, happiness_changes

voting_schemes = ['plurality', 'borda']
number_of_voters = 5
number_of_candidates = 4
group_sizes = [2,3,5]
num_trials = 2

# Run experiments
for voting_scheme in voting_schemes:
    for group_size in group_sizes:
        success_rate, avg_incentives, avg_happinesses, happiness_changes = run_experiments(voting_scheme, number_of_voters, number_of_candidates, group_size, num_trials)
        print(f"Voting Scheme: {voting_scheme}, Group Size: {group_size}")
        print(f"Collusion Success Rate: {success_rate:.2f}")
        print(f"Average Collusion Incentives: {avg_incentives}")
        print(f"Average Happinesses: {avg_happinesses}")
        print("")

         # Plot happiness changes
        plt.figure(figsize=(10, 6))
        for trial in range(num_trials):
            plt.plot(happiness_changes[trial], label=f'Trial {trial+1}')
        plt.xlabel('Voter Index')
        plt.ylabel('Happiness Level')
        plt.title(f'Happiness Level Changes - Voting Scheme: {voting_scheme}, Group Size: {group_size}')
        plt.legend()
        plt.show()


# print("=== Collusive Strategic Voting ===")
# btva_collusion = BTVA_Collusion('plurality', pref_matrix)
# collusive_incentives = btva_collusion.run_collusive_strategic_voting(group_size=4)


#print("Collusive Incentives:", collusive_incentives)

# print("\n=== Counter-Strategic Voting ===")
# btva_counter = BTVA_CounterStrategic('plurality', pref_matrix)
# counter_incentives = btva_counter.run_counter_strategic_voting(max_rounds=3)
# print("Counter-Strategic Incentives:", counter_incentives)

# print("\n=== Imperfect Information Strategic Voting ===")
# btva_imperfect = BTVA_ImperfectInfo('plurality', pref_matrix, noise_level=0.2)
# basic_result = btva_imperfect.run_non_strategic_election()
# imperfect_incentives = btva_imperfect.run_strategic_voting(basic_result)
# print("Imperfect Information Incentives:", imperfect_incentives)

# print("\n=== Concurrent Tactical Voting ===")
# btva_concurrent = BTVA_ConcurrentTactical('plurality', pref_matrix)
# concurrent_incentives = btva_concurrent.run_concurrent_tactical_voting()
# print("Concurrent Tactical Incentives:", concurrent_incentives)
