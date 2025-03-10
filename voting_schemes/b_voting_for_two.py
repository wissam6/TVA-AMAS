import numpy as np
from tvas.btva import BTVA
import itertools

class BVotingForTwo(BTVA):
    def run_non_strategic_election(self):
        scores = np.zeros(self.num_alternatives)
        for voter in range(self.num_voters):
            top_choices = self.preference_matrix[:2, voter]
            for choice in top_choices:
                scores[choice] += 1
        election_ranking = np.argsort(-scores, kind='stable')
        votes = np.sort(-scores, kind='stable').astype(int) * (-1)
        return np.vstack((election_ranking, votes))

    def run_strategic_election(self, election_result):
        # unlike borda, here we have O(n2) combinations of two alternatives, and it is feasible to check for
        # all of them to see if any would improve happiness.
        strategic_scenarios = [None] * self.num_voters
        for voter in range(self.num_voters):
            voter_preference = self.preference_matrix[:, voter]  
            original_happiness = self.non_strategic_happinesses[voter]
            voter_max_strategic_happiness = -1

            # checking for COMPROMISE/BURY possibilites
            for i, j in itertools.combinations(range(self.num_alternatives), 2):
                flagged_two_elements_array = [0] * self.num_alternatives
                flagged_two_elements_array[i] = 1
                flagged_two_elements_array[j] = 1

                # strategic elements (where the flag is 1)
                strategic_elements = [voter_preference[i] for i, flag in enumerate(flagged_two_elements_array) if flag == 1]
                # the remaining elements (flag is 0) with the original order
                remaining_elements = [voter_preference[i] for i, flag in enumerate(flagged_two_elements_array) if flag == 0]

                # output the two arrays where the two strategic alternatives are at the top
                for perm in itertools.permutations(strategic_elements):
                    potential_strategic_preference = list(perm) + remaining_elements
                    new_preference_matrix = np.ndarray.copy(self.preference_matrix)
                    new_preference_matrix[:, voter] = potential_strategic_preference
                    btva_strategic = BVotingForTwo(new_preference_matrix, self.happiness_function)
                    new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                    
                    new_happinesses = self.calc_happinesses(new_election_ranking)
                    if (new_happinesses[voter] > original_happiness) and (new_happinesses[voter] > voter_max_strategic_happiness):
                        voter_max_strategic_happiness = new_happinesses[voter]
                        strategic_scenarios[voter] = {
                            'strategy': 'compromise/bury',
                            'strategic preference matrix': new_preference_matrix,
                            'new election ranking': new_election_ranking,
                            'new votes': new_votes,
                            'new happinesses': new_happinesses,
                            'voter original happiness': original_happiness,
                            'voter strategic happiness': new_happinesses[voter]
                        }
                        
        return strategic_scenarios
