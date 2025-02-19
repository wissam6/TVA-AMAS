import numpy as np
import random
from happiness import *
import math

class BTVA:
    def __init__(self, voting_scheme, preference_matrix):
        self.voting_scheme = voting_scheme
        self.preference_matrix = preference_matrix
        ## Each row represents an alternative's rankings by different voters
        self.num_alternatives, self.num_voters = preference_matrix.shape
        self.risk_of_strategic_voting = 0
        self.non_strategic_happinesses = np.zeros(self.num_voters)


    def run_non_strategic_election(self):
        scores = np.zeros(self.num_alternatives)
        if self.voting_scheme == 'plurality':
            for voter in range(self.num_voters):
                top_choice = self.preference_matrix[0, voter]
                scores[top_choice] += 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)

        elif self.voting_scheme == 'voting_for_two':
            for voter in range(self.num_voters):
                top_choices = self.preference_matrix[:2, voter]
                for choice in top_choices:
                    scores[choice] += 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            
        elif self.voting_scheme == 'anti_plurality':
            scores.fill(self.num_voters)
            for voter in range(self.num_voters):
                last_choice = self.preference_matrix[-1, voter]
                scores[last_choice] -= 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            
        elif self.voting_scheme == 'borda':
            for voter in range(self.num_voters):
                for rank, choice in enumerate(self.preference_matrix[:, voter]):
                    if not np.isin(choice, range(self.num_alternatives)):
                        continue
                    else:
                        scores[choice] += (self.num_alternatives - rank - 1)
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
        
        election_result = np.vstack((election_ranking, votes))
        
        return election_result

    def calc_happinesses(self, preference_matrix, election_ranking, happiness_function, **kwargs):
        voters = np.arange(self.num_voters)
        happinesses = np.zeros(self.num_voters)
        for voter in voters:
            happinesses[voter] = happiness_function(preference_matrix[:, voter], election_ranking)
        return happinesses

    def run_strategic_voting(self, election_result):
        election_ranking, votes = election_result 

        if self.voting_scheme == 'plurality':
            winner = election_ranking[0]
            result_copy = np.ndarray.copy(election_result)
            result_copy[1, 0] = -1 # changing winner vote to -1 to find next max vote
            results_without_the_winner = result_copy
            second_max_vote = np.max(results_without_the_winner[1])
            contenders_indices = np.where(results_without_the_winner[1] == second_max_vote)[0]
            contenders = election_result[0, contenders_indices]

            strategic_voting_incentives = np.zeros(self.num_voters)

            for voter in range(self.num_voters):
                voter_preference = self.preference_matrix[:, voter]
                voter_rank_for_winner = np.where(voter_preference == winner)[0][0]
                
                for contender in contenders:
                    voter_rank_for_contender = np.where(voter_preference == contender)[0][0]
                    if (voter_rank_for_winner < voter_rank_for_contender):
                        # the voter doesn't have an incentive to vote strategically
                        continue
                    else:
                        other_alternatives = np.delete(voter_preference, [voter_rank_for_winner, voter_rank_for_contender])
                        # a simple tactic of comporomising and burying at the same time for maximizing the chance of success.
                        strategic_preference = np.concatenate(([contender], other_alternatives, [winner]))
                        strategic_preference_matrix = np.ndarray.copy(self.preference_matrix)
                        strategic_preference_matrix[:, voter] = strategic_preference
                        btva_strategic = BTVA('plurality', strategic_preference_matrix)
                        new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                        new_winner = new_election_ranking[0]
                        if new_winner != winner:
                            new_happinesses = self.calc_happinesses(self.preference_matrix, new_election_ranking, exponential_decay_happiness)
                            print(f'Voter {voter} strategic voting >>>')
                            print('New Preference Matrix')
                            print(strategic_preference_matrix)
                            print('New Winner', new_winner)
                            print('New Election Ranking', new_election_ranking)
                            print('New Election Scores ', new_votes)
                            print("Original Voters' Happiness", np.array2string(self.non_strategic_happinesses, formatter={'float_kind': lambda x: f"{x:.2f}"}))
                            print("Updated Voters' Happiness ", np.array2string(new_happinesses, formatter={'float_kind': lambda x: f"{x:.2f}"}))
                            print()

                            if strategic_voting_incentives[voter] == 0:
                                strategic_voting_incentives[voter] = 1 
            self.risk_of_strategic_voting = np.sum(strategic_voting_incentives) / self.num_voters

        elif self.voting_scheme == 'borda':
            # election_ranking, votes = election_result 
            strategic_voting_incentives = np.zeros(self.num_voters)
            winner  = election_ranking[0]
            contenders = election_ranking[1:]

            for voter in range(self.num_voters):
                voter_preference = self.preference_matrix[:, voter]                
                # in borda any change can potentially increase happiness, so every alternative is a contender
                for contender in contenders:
                    # 1st strategy: bullet voting
                    strategic_preference = np.full_like(voter_preference, -1)
                    strategic_preference[0] = contender
                    strategic_preference_matrix = np.ndarray.copy(self.preference_matrix)
                    strategic_preference_matrix[:, voter] = strategic_preference
                    btva_strategic = BTVA('borda', strategic_preference_matrix)
                    new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                    
                    new_happinesses = self.calc_happinesses(self.preference_matrix, new_election_ranking, exp_decay_borda_style_happiness)

                    if new_happinesses[voter] > self.non_strategic_happinesses[voter]:
                        print(f'Voter {voter} strategic voting >>>')
                        print('New Preference Matrix')
                        print(strategic_preference_matrix)
                        print('New Election Ranking', new_election_ranking)
                        print('New Election Scores ', new_votes)
                        print("Original Voters' Happiness", np.array2string(self.non_strategic_happinesses, formatter={'float_kind': lambda x: f"{x:.2f}"}))
                        print("Updated Voters' Happiness ", np.array2string(new_happinesses, formatter={'float_kind': lambda x: f"{x:.2f}"}))
                        print()
                    
                        
                        if strategic_voting_incentives[voter] == 0:
                            strategic_voting_incentives[voter] = 1

            self.risk_of_strategic_voting = np.sum(strategic_voting_incentives) / self.num_voters

            # compromise
            # bury

        return strategic_voting_incentives


pref_matrix = np.array([
    [0, 1, 3, 0, 2, 3, 2, 3],
    [2, 0, 0, 1, 4, 4, 4, 2],
    [1, 3, 2, 2, 1, 0, 0, 1],
    [3, 4, 4, 3, 3, 1, 1, 4],
    [4, 2, 1, 4, 0, 2, 3, 0]
])

# Example Usage
btva = BTVA('borda', pref_matrix)

# Non-strategic election
election_result = btva.run_non_strategic_election()
election_ranking, votes = election_result
happinesses = btva.calc_happinesses(pref_matrix, election_ranking, exp_decay_borda_style_happiness)
btva.non_strategic_happinesses = happinesses
print()
print('Non-strategic Election >>>')
print('Winner', election_ranking[0])
print('Election Ranking', election_ranking)
print('Election Scores ', votes)
print("Voters' Happinesses ", np.array2string(happinesses, formatter={'float_kind': lambda x: f"{x:.2f}"}))
print()

# Strategic election
strategic_voting_incentives = btva.run_strategic_voting(election_result)
print('Risk of Strategic Voting:', btva.risk_of_strategic_voting)
print('Potential Strategic Voters:', strategic_voting_incentives.astype(int))