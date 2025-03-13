import numpy as np
from btva import BTVA

class BBorda(BTVA):
    def run_non_strategic_election(self):
        scores = np.zeros(self.num_alternatives)
        for voter in range(self.num_voters):
            for rank, choice in enumerate(self.preference_matrix[:, voter]):
                if not np.isin(choice, range(self.num_alternatives)):
                    continue
                scores[choice] += (self.num_alternatives - rank - 1)
        election_ranking = np.argsort(-scores, kind='stable')
        votes = np.sort(-scores, kind='stable').astype(int) * (-1)
        return np.vstack((election_ranking, votes))
    
    def run_strategic_election(self, election_result):
        election_ranking, votes = election_result
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
                        btva_strategic = BBorda(strategic_preference_matrix, self.happiness_function)
                        new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                        
                        new_happinesses = self.calc_happinesses(new_election_ranking)

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
                    #     print(f"{voter_preference} -> {best_bullet_scenarios[voter]['strategic preference matrix'][:, voter]}")
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
                            
                            btva_strategic = BBorda(strategic_preference_matrix, self.happiness_function)
                            new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                            new_happinesses = self.calc_happinesses(new_election_ranking)
                            
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
                    #     print(f"{voter_preference} -> {best_compromise_scenarios[voter]['strategic preference matrix'][:, voter]}")
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
                                                
                            btva_strategic = BBorda(strategic_preference_matrix, self.happiness_function)
                            new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                            new_happinesses = self.calc_happinesses(new_election_ranking)
                            
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
                    #     print(f"{voter_preference} -> {best_bury_scenarios[voter]['strategic preference matrix'][:, voter]}")
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
