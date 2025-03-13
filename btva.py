import numpy as np

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

    def pretty_print_scenarios(self, strategic_scenarios):
        for voter in range(self.num_voters):
            if strategic_scenarios[voter]:
                print(f"VOTER {voter} {strategic_scenarios[voter]['strategy'].upper()}ING >>>")
                print(f"{self.preference_matrix[:, voter]} -> {strategic_scenarios[voter]['strategic preference matrix'][:, voter]}")
                for key, value in strategic_scenarios[voter].items():
                    if key == 'strategic preference matrix':
                        print(key, value, sep='\n')
                    elif key == 'strategy':
                        pass
                    else:
                        print(f'{key}: {value}')
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