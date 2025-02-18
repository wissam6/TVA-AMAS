import numpy as np
import random
from happiness import *
import math
import json

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
            strategic_voting_incentives = np.zeros(self.num_voters)
            winner  = election_ranking[0]
            strategies = ['bullet', 'compromise', 'bury'] 

            best_strategic_scenarios = [None] * self.num_voters
            for voter in range(self.num_voters):
                voter_preference = self.preference_matrix[:, voter]  
                original_happiness = self.non_strategic_happinesses[voter]
                
                voter_max_bullet_happiness = -1
                voter_max_compromise_happiness = -1
                voter_max_bury_happiness = -1

                for strategy in strategies:

                    #BULLET
                    if strategy == 'bullet':
                        best_bullet_scenarios = [None] * self.num_voters

                        for contender in range(self.num_alternatives):
                            strategic_preference = np.full_like(voter_preference, -1)
                            strategic_preference[0] = contender
                            strategic_preference_matrix = np.ndarray.copy(self.preference_matrix)
                            strategic_preference_matrix[:, voter] = strategic_preference
                            btva_strategic = BTVA('borda', strategic_preference_matrix)
                            new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                            
                            new_happinesses = self.calc_happinesses(self.preference_matrix, new_election_ranking, exp_decay_borda_style_happiness)

                            if (new_happinesses[voter] > original_happiness) and (new_happinesses[voter] > voter_max_bullet_happiness):
                                voter_max_bullet_happiness = new_happinesses[voter]
                                
                                best_bullet_scenarios[voter] = {
                                    'strategy': 'bullet',
                                    'strategic preference matrix': strategic_preference_matrix,
                                    'new election ranking': new_election_ranking,
                                    'new votes': new_votes,
                                    'new happinesses': new_happinesses,
                                    'voter original happiness': original_happiness,
                                    'voter strategic happiness': voter_max_bullet_happiness
                                }

                        # # PRINTING BULLET SCENARIOS
                        # if best_bullet_scenarios[voter]:
                        #     print(f'VOTER {voter} STRATEGIC VOTING >>>')
                        #     print(f'{voter_preference} -> {best_bullet_scenarios[voter]['strategic preference matrix'][:, voter]}')
                        #     for key, value in best_bullet_scenarios[voter].items():
                        #         if key == 'strategic preference matrix':
                        #             print(key, value, sep='\n')
                        #         else:
                        #             print(f'{key}: {value}')
                        #     print()

                    #COMPROMISE
                    elif strategy == 'compromise':
                        best_compromise_scenarios = [None] * self.num_voters

                        for contender in election_ranking[1:]:
                            # Skip if the contender is already the voter's first preference, since compromising it wouldn't change anything
                            if contender == voter_preference[0]:
                                continue
                            
                            original_index = np.where(voter_preference == contender)[0][0]

                            for i in range(1, original_index + 1):
                                strategic_preference = np.copy(voter_preference)
                                strategic_preference = np.delete(strategic_preference, original_index)
                                # sliding up the contender 1 place at a time
                                strategic_preference = np.insert(strategic_preference, original_index - i, contender)
                                
                                strategic_preference_matrix = np.copy(self.preference_matrix)
                                strategic_preference_matrix[:, voter] = strategic_preference
                                
                                btva_strategic = BTVA('borda', strategic_preference_matrix)
                                new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                                new_happinesses = self.calc_happinesses(self.preference_matrix, new_election_ranking, exp_decay_borda_style_happiness)
                                
                                # Check if this strategic move increases the voter's happiness
                                if (new_happinesses[voter] > original_happiness) and (new_happinesses[voter] > voter_max_compromise_happiness):
                                    
                                    voter_max_compromise_happiness = new_happinesses[voter]
                                    
                                    best_compromise_scenarios[voter] = {
                                        'strategy': 'compromise',
                                        'strategic preference matrix': strategic_preference_matrix,
                                        'new election ranking': new_election_ranking,
                                        'new votes': new_votes,
                                        'new happinesses': new_happinesses,
                                        'voter original happiness': original_happiness,
                                        'voter strategic happiness': voter_max_compromise_happiness
                                    }
                        
                        # # PRINTING COMPOROMISE SCENARIOS
                        # if best_compromise_scenarios[voter]:
                        #     print(f'VOTER {voter} STRATEGIC VOTING >>>')
                        #     print(f'{voter_preference} -> {best_compromise_scenarios[voter]['strategic preference matrix'][:, voter]}')
                        #     for key, value in best_compromise_scenarios[voter].items():
                        #         if key == 'strategic preference matrix':
                        #             print(key, value, sep='\n')
                        #         else:
                        #             print(f'{key}: {value}')
                        #     print()

                    #BURY
                    elif strategy == 'bury':
                        best_bury_scenarios = [None] * self.num_voters

                        for contender in election_ranking[1:]:
                            # Skip if the contender is already the last preference, since burying it wouldn't change anything
                            if contender == voter_preference[-1]:
                                continue
                            
                            original_index = np.where(voter_preference == contender)[0][0]

                            for i in range(1, self.num_alternatives - original_index):
                                strategic_preference = np.copy(voter_preference)
                                strategic_preference = np.delete(strategic_preference, original_index)
                                strategic_preference = np.insert(strategic_preference, original_index + i, contender)

                                strategic_preference_matrix = np.copy(self.preference_matrix)
                                strategic_preference_matrix[:, voter] = strategic_preference
                                                    
                                btva_strategic = BTVA('borda', strategic_preference_matrix)
                                new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                                new_happinesses = self.calc_happinesses(self.preference_matrix, new_election_ranking, exp_decay_borda_style_happiness)
                                
                                # Check if this strategic move increases the voter's happiness
                                if (new_happinesses[voter] > original_happiness) and (new_happinesses[voter] > voter_max_bury_happiness):
                                    
                                    voter_max_bury_happiness = new_happinesses[voter]
                                    
                                    best_bury_scenarios[voter] = {
                                        'strategy': 'bury', 
                                        'strategic preference matrix': strategic_preference_matrix,
                                        'new election ranking': new_election_ranking,
                                        'new votes': new_votes,
                                        'new happinesses': new_happinesses,
                                        'voter original happiness': original_happiness,
                                        'voter strategic happiness': voter_max_bury_happiness
                                    }

                        # # PRINTING BURY SCENARIOS    
                        # if best_bury_scenarios[voter]:
                        #     print(f'VOTER {voter} STRATEGIC VOTING >>>')
                        #     print(f'{voter_preference} -> {best_bury_scenarios[voter]['strategic preference matrix'][:, voter]}')
                        #     for key, value in best_bury_scenarios[voter].items():
                        #         if key == 'strategic preference matrix':
                        #             print(key, value, sep='\n')
                        #         else:
                        #             print(f'{key}: {value}')
                        #     print()

                # Finding the best strategic scenarios
                strategic_scenarios = [best_bullet_scenarios[voter], best_compromise_scenarios[voter], best_bury_scenarios[voter]]
                if all(value is None for value in strategic_scenarios):
                    # let best_strategic_scenarios[voter] remain None
                    pass
                else:
                    max_strategic_happiness = 0
                    for scenario in strategic_scenarios:
                        if scenario and (scenario['voter strategic happiness'] > max_strategic_happiness):
                            best_strategic_scenarios[voter] = scenario

        return best_strategic_scenarios
    
## FUNCTION TO GENERATE RANDOM PREFERENCE MATRICES   
def generate_random_preferences_matrix(num_alternatives, num_voters):
    return np.array([np.random.permutation(num_alternatives) for _ in range(num_voters)]).T

pref_matrix = np.array([
    [1, 1, 3, 3, 2, 2, 3, 2],
    [4, 3, 2, 0, 0, 3, 0, 3],
    [3, 0, 4, 4, 1, 4, 4, 0],
    [0, 4, 1, 2, 4, 1, 2, 4],
    [2, 2, 0, 1, 3, 0, 1, 1]
])

random_matrix = generate_random_preferences_matrix(num_alternatives=5, num_voters=8)

# Example Usage
btva = BTVA('borda', random_matrix)

## Non-strategic election
election_result = btva.run_non_strategic_election()
election_ranking, votes = election_result
happinesses = btva.calc_happinesses(random_matrix, election_ranking, exp_decay_borda_style_happiness)
btva.non_strategic_happinesses = happinesses
print()
print(random_matrix)
print()
print(f'NON-STRATEGIC ELECTION ({btva.voting_scheme.upper()}) >>>')
print('Winner', election_ranking[0])
print('Election Ranking', election_ranking)
print('Election Scores ', votes)
print("Voters' Happiness ", np.array2string(happinesses, formatter={'float_kind': lambda x: f"{x:.2f}"}))
print()

## Strategic election
best_strategic_scenarios = btva.run_strategic_voting(election_result)
# PRINTING BEST STRATEGIC SCENARIOS    
for voter in range(btva.num_voters):
    if best_strategic_scenarios[voter]:
        print(f'VOTER {voter} {best_strategic_scenarios[voter]['strategy'].upper()}ING >>>')
        print(f'{btva.preference_matrix[:, voter]} -> {best_strategic_scenarios[voter]['strategic preference matrix'][:, voter]}')
        for key, value in best_strategic_scenarios[voter].items():
            if key == 'strategic preference matrix':
                print(key, value, sep='\n')
            elif key == 'strategy':
                pass
            else:
                print(f'{key}: {value}')
        print()

voter_strategic_gains = np.zeros(btva.num_voters)
for voter in range(btva.num_voters):
    if not best_strategic_scenarios[voter]:
        pass
    else:
        voter_strategic_gains[voter] = (best_strategic_scenarios[voter]['voter strategic happiness'] - best_strategic_scenarios[voter]['voter original happiness']) / best_strategic_scenarios[voter]['voter strategic happiness']

print('VOTER STRATEGIC HAPPINESS GAINS:')
print(np.array2string(voter_strategic_gains * 100, formatter={'float_kind': lambda x: f"{x:.0f}%"}))