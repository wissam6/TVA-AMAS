import numpy as np
from tvas.btva import BTVA

class BPlurality(BTVA):
    def run_non_strategic_election(self):
        scores = np.zeros(self.num_alternatives)
        for voter in range(self.num_voters):
            top_choice = self.preference_matrix[0, voter]
            scores[top_choice] += 1
        election_ranking = np.argsort(-scores, kind='stable')
        votes = np.sort(-scores, kind='stable').astype(int) * (-1)
        return np.vstack((election_ranking, votes))
    
    def run_strategic_election(self, election_result):
        election_ranking, votes = election_result
        winner = election_ranking[0]
        result_copy = np.copy(election_result)
        result_copy[1, 0] = -1  # Changing winner's vote to -1 to find next max vote
        second_max_vote = np.max(result_copy[1])
        contenders_indices = np.where(result_copy[1] == second_max_vote)[0]
        contenders = election_result[0, contenders_indices]
        
        strategic_scenarios = [None] * self.num_voters
        for voter in range(self.num_voters):
            voter_preference = self.preference_matrix[:, voter]
            voter_rank_for_winner = np.where(voter_preference == winner)[0][0]
            for contender in contenders:
                voter_rank_for_contender = np.where(voter_preference == contender)[0][0]
                if voter_rank_for_winner < voter_rank_for_contender:
                    continue  # no incentive if winner is already preferred
                else:
                    # create a strategic preference: push down the winner
                    other_alternatives = np.delete(voter_preference, [voter_rank_for_winner, voter_rank_for_contender])
                    strategic_preference = np.concatenate(([contender], other_alternatives, [winner]))
                    strategic_preference_matrix = np.copy(self.preference_matrix)
                    strategic_preference_matrix[:, voter] = strategic_preference
                    # Run a new non-strategic election with the modified preferences
                    btva_strategic = BPlurality(strategic_preference_matrix, self.happiness_function)
                    new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                    new_winner = new_election_ranking[0]
                    new_happinesses = self.calc_happinesses(new_election_ranking)

                    if new_happinesses[voter] > self.non_strategic_happinesses[voter]:
                        strategic_scenarios[voter] = {
                                'strategy': 'compromise/bury',
                                'strategic preference matrix': strategic_preference_matrix,
                                'new election ranking': new_election_ranking,
                                'new votes': new_votes,
                                'new happinesses': new_happinesses,
                                'voter original happiness': self.non_strategic_happinesses[voter],
                                'voter strategic happiness': new_happinesses[voter]
                            }

        return strategic_scenarios
