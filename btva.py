import numpy as np
from helper_functions import print_side_by_side

class BTVA:
    def __init__(self, preference_matrix, happiness_function):
        self.preference_matrix = preference_matrix
        self.num_alternatives, self.num_voters = preference_matrix.shape
        self.non_strategic_happinesses = np.zeros(self.num_voters)
        self.happiness_function = happiness_function
    
    def run_non_strategic_election(self):
        pass
    
    def run_strategic_election(self, election_result):
        pass
    
    def calc_happinesses(self, election_ranking, preference_matrix=None):
        happinesses = np.zeros(self.num_voters)
        if preference_matrix is None:
            preference_matrix = self.preference_matrix
        for voter in range(self.num_voters):
            happinesses[voter] = self.happiness_function(preference_matrix[:, voter], election_ranking)
        return happinesses

    def pretty_print_scenarios(self, strategic_scenarios, election_result):
        election_ranking, votes = election_result
        strategic_voters = [voter for voter, strategy in enumerate(strategic_scenarios) if strategy]
        if len(strategic_voters) == 0:
            print("There are no strategic voters in this election!")
        else:
            print(f"Strategic voters are -> {', '.join(str(v) for v in strategic_voters)}")
        print()
        for voter in range(self.num_voters):
            if strategic_scenarios[voter]:
                strategic_preference_matrix = strategic_scenarios[voter]['strategic preference matrix']
                new_election_ranking = strategic_scenarios[voter]['new election ranking']
                new_votes = strategic_scenarios[voter]['new votes']
                new_happinesses = strategic_scenarios[voter]['new happinesses']

                print(f":: VOTER {voter} {strategic_scenarios[voter]['strategy'].upper()}ING..")
                print_side_by_side(self.preference_matrix, strategic_preference_matrix)

                print(f"Winner: {election_ranking[0]} -> {new_election_ranking[0]}")
                print(f"Election Ranking: {election_ranking} -> {new_election_ranking}")
                print(f"Election Scores: {votes} -> {new_votes}")
                print(f"Original Happiness:  {self.non_strategic_happinesses}")
                print(f"Strategic Happiness: {new_happinesses}")
                change_in_happiness = np.round((new_happinesses - self.non_strategic_happinesses) * 100/ np.maximum(new_happinesses, self.non_strategic_happinesses)).astype(int)
                print(f"Change In Happiness: {change_in_happiness} %")
                print()
        return

    def calc_strategic_gains(self, strategic_scenarios):
        voter_strategic_gains = np.zeros(self.num_voters)
        for voter in range(self.num_voters):
            if not strategic_scenarios[voter]:
                pass
            else:
                strategic_happiness = strategic_scenarios[voter]['voter strategic happiness']
                original_happiness = strategic_scenarios[voter]['voter original happiness']
                voter_strategic_gains[voter] = (strategic_happiness - original_happiness) / strategic_happiness
        return voter_strategic_gains