import numpy as np
from btva import BTVA

class BAntiPlurality(BTVA):

    def run_non_strategic_election(self):
        scores = np.full(self.num_alternatives, self.num_voters)
        for voter in range(self.num_voters):
            last_choice = self.preference_matrix[-1, voter]
            scores[last_choice] -= 1
        election_ranking = np.argsort(-scores, kind='stable')
        votes = np.sort(-scores, kind='stable').astype(int) * (-1)
        return np.vstack((election_ranking, votes))
    
    
    def run_strategic_election(self, election_result):
        # in anti-plurality the voter does not have any options to push the disliked candidate lower
        # but can at least try to help a more favorable contender to win (like the case in plurality).
        # we could have also treated anti-plurality like borda and considered minimal gains for voters
        # from various compromises, burys, or bullet voting.
        election_ranking, votes = election_result
        winner = election_ranking[0]

        votes_copy = np.copy(votes)
        votes_copy[0] = -1  # Changing winner's vote to -1 to find next max vote
        second_max_vote = np.max(votes_copy)

        contenders_indices = np.where(votes_copy == second_max_vote)[0]
        contenders = election_ranking[contenders_indices]
        
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
                    btva_strategic = BAntiPlurality(strategic_preference_matrix, self.happiness_function)
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
