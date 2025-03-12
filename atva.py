import numpy as np
import itertools
from happiness import *
from btva import BTVA

###############################################################################
# 1. Advanced Voting System: Addressing Voter Collusion (Multiple-Voter Manipulations)
###############################################################################

# make it work for diff group sizes and maybe dont do it in case the voter is already satisfied.
# it currently uses only compromising/burying
class BTVA_Collusion(BTVA):
    def run_collusive_strategic_voting(self, max_group_size=2):
        """
        Check if groups of voters (up to max_group_size) can coordinate
        their tactical vote to change the election outcome.
        """
        # Run the baseline election with voters' true (non-strategic) preferences
        original_result = self.run_non_strategic_election()
        # Identify the original winner (the candidate ranked first in the results)
        original_winner = original_result[0, 0]
        # Initialize an array to track which voters would benefit from collusion
        collusion_incentives = np.zeros(self.num_voters)
        
        # Loop over all possible colluding group sizes (from 2 up to max_group_size)
        for group_size in range(2, max_group_size + 1):
            # Generate all combinations of voters of the current group size
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
                    # 3. Finally, the original winner at the bottomf
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


# ###############################################################################
# # 2. Advanced Voting System: Incorporating Counter-Strategic Voting
# ###############################################################################
# class BTVA_CounterStrategic(BTVA):
#     def run_counter_strategic_voting(self, max_rounds=3):
#         """
#         Iteratively allow voters to adjust their ballots in response to
#         observed strategic moves (counter-strategic voting).
#         """
#         current_pref_matrix = np.copy(self.preference_matrix)
#         round_num = 0
#         overall_incentives = np.zeros(self.num_voters)
        
#         while round_num < max_rounds:
#             print(f"Round {round_num+1} of counter-strategic voting:")
#             btva_instance = BTVA(self.voting_scheme, current_pref_matrix)
#             election_result = btva_instance.run_non_strategic_election()
#             original_winner = election_result[0, 0]
#             round_incentives = np.zeros(self.num_voters)
            
#             # For each voter, try a simple counter-move (swap top two positions)
#             for voter in range(self.num_voters):
#                 voter_pref = current_pref_matrix[:, voter]
#                 # Only consider counter-moves if the voter currently votes for the winner
#                 if voter_pref[0] != original_winner:
#                     continue
#                 new_ballot = np.copy(voter_pref)
#                 new_ballot[0], new_ballot[1] = new_ballot[1], new_ballot[0]
#                 new_pref_matrix = np.copy(current_pref_matrix)
#                 new_pref_matrix[:, voter] = new_ballot
#                 test_btva = BTVA(self.voting_scheme, new_pref_matrix)
#                 new_result = test_btva.run_non_strategic_election()
#                 new_winner = new_result[0, 0]
#                 if new_winner != original_winner:
#                     round_incentives[voter] = 1
#                     current_pref_matrix[:, voter] = new_ballot
#                     print(f"Voter {voter} counter-changed ballot to: {new_ballot} resulting in new winner {new_winner}")
#             if np.sum(round_incentives) == 0:
#                 break
#             overall_incentives += round_incentives
#             round_num += 1

#         return overall_incentives

# ###############################################################################
# # 3. Advanced Voting System: Imperfect Information about Voter Preferences
# ###############################################################################
# class BTVA_ImperfectInfo(BTVA):
#     def __init__(self, voting_scheme, preference_matrix, noise_level=0.1):
#         super().__init__(voting_scheme, preference_matrix)
#         self.noise_level = noise_level
#         # Generate a noisy version of preferences to simulate imperfect information.
#         self.noisy_preference_matrix = self.generate_noisy_preferences()

#     def generate_noisy_preferences(self):
#         """
#         Create a noisy version of the true preference matrix by randomly swapping
#         some candidates with probability defined by noise_level.
#         """
#         noisy_matrix = np.copy(self.preference_matrix)
#         num_alternatives, num_voters = noisy_matrix.shape
#         for voter in range(num_voters):
#             for i in range(num_alternatives):
#                 if random.random() < self.noise_level:
#                     swap_index = random.randint(0, num_alternatives - 1)
#                     noisy_matrix[i, voter], noisy_matrix[swap_index, voter] = noisy_matrix[swap_index, voter], noisy_matrix[i, voter]
#         return noisy_matrix

#     def run_strategic_voting(self, election_result):
#         """
#         Override strategic voting to use the noisy (imperfect) preferences.
#         """
#         election_ranking, votes = election_result 
#         winner = election_ranking[0]
#         incentives = np.zeros(self.num_voters)
#         for voter in range(self.num_voters):
#             voter_noisy_pref = self.noisy_preference_matrix[:, voter]
#             voter_rank_for_winner = np.where(voter_noisy_pref == winner)[0][0]
#             # Choose the best contender from the noisy ranking that is not the winner
#             potential_contenders = [c for c in voter_noisy_pref if c != winner]
#             if not potential_contenders:
#                 continue
#             contender = potential_contenders[0]
#             voter_rank_for_contender = np.where(voter_noisy_pref == contender)[0][0]
#             if voter_rank_for_winner > voter_rank_for_contender:
#                 strategic_ballot = np.concatenate(([contender], np.delete(voter_noisy_pref, np.where(voter_noisy_pref == contender))))
#                 new_pref_matrix = np.copy(self.preference_matrix)
#                 new_pref_matrix[:, voter] = strategic_ballot
#                 test_btva = BTVA(self.voting_scheme, new_pref_matrix)
#                 new_result = test_btva.run_non_strategic_election()
#                 new_winner = new_result[0, 0]
#                 if new_winner != winner:
#                     incentives[voter] = 1
#                     print(f"Voter {voter} (imperfect info) changed ballot to: {strategic_ballot} resulting in new winner {new_winner}")
#         return incentives

# ###############################################################################
# # 4. Advanced Voting System: Concurrent Tactical Voting (Multiple Tactical Moves)
# ###############################################################################
# class BTVA_ConcurrentTactical(BTVA):
#     def run_concurrent_tactical_voting(self):
#         """
#         Allow each voter to try multiple tactical ballots concurrently and adopt
#         the one that most improves their personal outcome.
#         """
#         original_result = self.run_non_strategic_election()
#         original_winner = original_result[0, 0]
#         best_incentives = np.zeros(self.num_voters)
#         best_happiness = np.zeros(self.num_voters)
#         new_pref_matrix = np.copy(self.preference_matrix)
        
#         # For each voter, try different reorderings of the top three positions.
#         for voter in range(self.num_voters):
#             voter_pref = self.preference_matrix[:, voter]
#             best_ballot = voter_pref.copy()
#             # Generate all permutations of the first three candidates.
#             for perm in itertools.permutations(voter_pref[:3]):
#                 candidate_ballot = np.array(list(perm) + list(voter_pref[3:]))
#                 test_matrix = np.copy(new_pref_matrix)
#                 test_matrix[:, voter] = candidate_ballot
#                 test_btva = BTVA(self.voting_scheme, test_matrix)
#                 test_result = test_btva.run_non_strategic_election()
#                 new_winner = test_result[0, 0]
#                 # Calculate new happiness (using an example happiness function).
#                 #new_happiness = hpns.exponential_decay_happiness(voter_pref, test_result[0])
#                 new_happiness = exponential_decay_happiness(voter_pref, test_result[0])
#                 if new_happiness > best_happiness[voter] and new_winner != original_winner:
#                     best_happiness[voter] = new_happiness
#                     best_ballot = candidate_ballot
#                     best_incentives[voter] = 1
#             new_pref_matrix[:, voter] = best_ballot
        
#         # Re-run the election with all updated ballots.
#         final_btva = BTVA(self.voting_scheme, new_pref_matrix)
#         final_result = final_btva.run_non_strategic_election()
#         final_winner = final_result[0, 0]
#         print("Final Winner after concurrent tactical voting:", final_winner)

#         return best_incentives

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

print("=== Collusive Strategic Voting ===")
btva_collusion = BTVA_Collusion('plurality', pref_matrix)
collusive_incentives = btva_collusion.run_collusive_strategic_voting(max_group_size=2)
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
