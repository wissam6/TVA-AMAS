import numpy as np
import itertools
from happiness import *
from btva import BTVA

# class BTVA_ConcurrentTactical(BTVA):
def run_concurrent_tactical_voting(self, max_group_size):
    # randomized of tactical voting
    # BTVA: strategic collection, i
    #loosing ones do tactical voting and we know them, calculate all calculations of them, 2^ n= 5 voter becomes 32
    # subset votes strategically, perfect information -> no one knows anything, thats the catch, thats the difference
    # stratetic voting, refacture branch -> tactical voting
    # ignore btva in root, use tvas/btva as a new file
#         """
#         Allow each voter to try multiple tactical ballots concurrently and adopt
#         the one that most improves their personal outcome.
#         """

    # from voting collusion:
    # Run the baseline election with voters' true (non-strategic) preferences
    original_result = self.run_non_strategic_election()
    # Identify the original winner (the candidate ranked first in the results)
    original_winner = original_result[0, 0]
    # Initialize an array to track which voters would benefit from collusion
    collusion_incentives = np.zeros(self.num_voters)
        
    # Loop over all possible number of strategic voter (from 2 up to max_group_size)
    for group_size in range(2, max_group_size + 1):
        # Generate all combinations of voters of the current group size
        for group in itertools.combinations(range(self.num_voters), group_size):
            # Create a copy of the original preference matrix to modify ballots for this group
            new_pref_matrix = np.copy(self.preference_matrix)
            # Dictionary to hold cumulative scores for each candidate (excluding the original winner)
           # candidate_scores = {}
            # Evaluate each voter in the group
            for voter in group:
                voter_scores ={}
                # Collect for each voter it's best candidate, independent of the choices of the other voters candidates in the group 
                # -> imperfect information
                tactical_candidates = {} 
                # For each candidate in the voter's ranking
                for candidate in self.preference_matrix[:, voter]:
                    if candidate != original_winner:
                        # Find the candidate's rank (position) in the voter's ballot
                        rank = np.where(self.preference_matrix[:, voter] == candidate)[0][0]
                        # Sum the rank scores over the group; lower score indicates a more preferred candidate
                        voter_scores[candidate] = rank
                        
                # If no candidate was found (unlikely), skip this group
                # if not candidate_scores:
                #    continue
                # Choose the candidate with the lowest total rank as the collusive candidate
                tactical_candidates[voter] = min(voter_scores, key=voter_scores.get)
                    
                # For each voter in the group of tactical voters, modify their ballot:
                # Place the collusive candidate first and the original winner last.
                for voter in group:
                    voter_pref = self.preference_matrix[:, voter]
                    # Build a new ballot:
                    # 1. The candidate of the current voter at the top
                    # 2. Then all other candidates (preserving their order) except the candidate of the current voter and original winner
                    # 3. Finally, the original winner at the bottomf
                    new_ballot = [tactical_candidates[voter]] + [c for c in voter_pref if c not in [tactical_candidates[voter], original_winner]] + [original_winner]
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

