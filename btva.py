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
            print("Non-strategic voters -> " + ", ".join(str(v) for v in range(self.num_voters) if v not in strategic_voters))
            print(f"Strategic voters are -> {', '.join(str(v) for v in strategic_voters)}")
            print()
            print("=" * 24)
            print("Best strategic scenarios")
            print("=" * 24)
        print()
        for voter in range(self.num_voters):
            if strategic_scenarios[voter]:
                strategic_preference_matrix = strategic_scenarios[voter]['strategic preference matrix']
                new_election_ranking = strategic_scenarios[voter]['new election ranking']
                new_votes = strategic_scenarios[voter]['new votes']
                new_happinesses = strategic_scenarios[voter]['new happinesses']

                print()
                print(f"..VOTER {voter} {strategic_scenarios[voter]['strategy'].upper()}ING..")
                print(f"{self.preference_matrix[:, voter]} -> {strategic_preference_matrix[:, voter]}")
                print_side_by_side(self.preference_matrix, strategic_preference_matrix)

                print("Change in Voting Outcome (O):")
                print(f"Winner: {election_ranking[0]} -> {new_election_ranking[0]}")
                print(f"Ranking: {election_ranking} -> {new_election_ranking}")
                print(f"Votes: {votes} -> {new_votes}")
                print(f"Original Happiness (Hi):  {self.non_strategic_happinesses}")
                print(f"Strategic Happiness (H~i): {new_happinesses}")
                change_in_happiness = np.round((new_happinesses - self.non_strategic_happinesses) * 100/ np.maximum(new_happinesses, self.non_strategic_happinesses)).astype(int)
                change_in_overall_happiness = np.round((sum(new_happinesses) - sum(self.non_strategic_happinesses)) * 100/ np.maximum(sum(new_happinesses), sum(self.non_strategic_happinesses))).astype(int)
                print(f"Overall Voter Happiness Level (H): {np.around(sum(self.non_strategic_happinesses), 2)} (Out of max possible {self.num_voters})")
                print(f"Strategic Voter Happiness Level (H~): {np.around(sum(new_happinesses), 2)} (Out of max possible {self.num_voters})")
                print(f"Change In Individual Happiness: {change_in_happiness} %")
                print(f"Change In Overall Happiness: {change_in_overall_happiness} %")

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