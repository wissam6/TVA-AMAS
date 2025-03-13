import numpy as np
from btva import BTVA
from b_plurality import BPlurality
from b_anti_plurality import BAntiPlurality
from b_voting_for_two import BVotingForTwo
from b_borda import BBorda
from happiness import *
from risk import *
from functools import partial
import itertools

# strategic voting by multiple voters at the same time
class ATVA4(BTVA):
    def __init__(self, preference_matrix, voting_scheme):
        self.num_alternatives, self.num_voters = preference_matrix.shape
        self.voting_scheme = voting_scheme
        self.btva_happiness_functions_dict = {
            'plurality': exponential_decay_happiness,
            'anti_plurality': partial(
                exp_decay_borda_style_happiness,
                # assuming the voter wants the first preference to win, and the last preference to lose 
                polarization={'win_fr': 1/self.num_alternatives, 'lose_fr': 1/self.num_alternatives, 'wl_importance': 1}
            ),
            'voting_for_two': partial(
                exp_decay_borda_style_happiness,
                polarization={'win_fr': 2/self.num_alternatives, 'lose_fr': 0, 'wl_importance': 1}
            ),
            'borda': exp_decay_borda_style_happiness
        }
        super().__init__(preference_matrix, self.btva_happiness_functions_dict[self.voting_scheme])
    
    def print_side_by_side(self, A, B):
        # Convert A and B into their string representations line by line
        A_str = str(A).splitlines()
        B_str = str(B).splitlines()
        
        # Pad the shorter list so they have the same number of lines
        n_lines = max(len(A_str), len(B_str))
        A_str += [""] * (n_lines - len(A_str))
        B_str += [""] * (n_lines - len(B_str))
        
        # Compute the width of the left block for alignment
        left_width = max(len(line) for line in A_str)
        
        # Calculate the "middle" line
        mid_line_idx = n_lines // 2
        
        # Print line by line
        for i, (left_line, right_line) in enumerate(zip(A_str, B_str)):
            if i == mid_line_idx:
                # Only print " -> " in the middle line
                print(left_line.ljust(left_width) + " -> " + right_line)
            else:
                # Print spacing on non-middle lines
                print(left_line.ljust(left_width) + "    " + right_line)


    def run_strategic_btva_election(self):
        btva_classes_dict = {
            'plurality': BPlurality,
            'anti_plurality': BAntiPlurality,
            'voting_for_two': BVotingForTwo,
            'borda': BBorda
        }

        # creating a btva to run a strategic election and see which voters want to vote strategically and in what scenarios
        btva_instance = btva_classes_dict[self.voting_scheme](self.preference_matrix, self.btva_happiness_functions_dict[self.voting_scheme])

        # Running a non-strategic btva election
        election_result = btva_instance.run_non_strategic_election()
        election_ranking, votes = election_result
        happinesses = btva_instance.calc_happinesses(election_ranking)
        btva_instance.non_strategic_happinesses = happinesses

        print("Preference Matrix:")
        print(self.preference_matrix)
        print()
        print(f"NON-STRATEGIC {self.voting_scheme.upper().replace('_', '-')} ELECTION >>>")
        print("Winner:", election_ranking[0])
        print("Election Ranking:", election_ranking)
        print("Election Scores:", votes)
        print("Voters' Happiness:", happinesses)
        print()

        # Running a strategic btva election
        strategic_scenarios = btva_instance.run_strategic_election(election_result)

        # Running a strategic atva election for every combination of 2 or more strategic voters
        strategic_voters = [voter for voter, strategy in enumerate(strategic_scenarios) if strategy]
        if len(strategic_voters) == 0:
            print("There are no strategic voters in this election!")
        elif len(strategic_voters) == 1:
            print(f"The only strategic voter is {strategic_voters[0]}; No concurrent voting possible.")
        else:
            print(f"Strategic voters are -> {', '.join(str(v) for v in strategic_voters)}")
            
        print()

        for n in range(2, len(strategic_voters) + 1):
            print("=" * 33)
            print(f"# {n} voters voting simultaneously")
            print("=" * 33)
            for combo in itertools.combinations(strategic_voters, n):
                print(f":: voters {', '.join(str(v) for v in combo)} simultaneously voting strategically..")
                strategic_preference_matrix = np.copy(self.preference_matrix)
                combo = np.array(combo)
                new_pref_columns = np.column_stack(
                    [strategic_scenarios[voter]['strategic preference matrix'][:, voter] for voter in combo]
                )
                strategic_preference_matrix[:, combo] = new_pref_columns
                self.print_side_by_side(self.preference_matrix, strategic_preference_matrix)
                print()

                # creating a btva to run a strategic election and see which voters want to vote strategically and in what scenarios
                new_btva_instance = btva_classes_dict[self.voting_scheme](strategic_preference_matrix, self.btva_happiness_functions_dict[self.voting_scheme])

                # Running a non-strategic btva election
                new_election_result = new_btva_instance.run_non_strategic_election()
                new_election_ranking, new_votes = new_election_result
                new_happinesses = new_btva_instance.calc_happinesses(new_election_ranking, self.preference_matrix)
                new_btva_instance.non_strategic_happinesses = new_happinesses

                print(f"Winner: {election_ranking[0]} -> {new_election_ranking[0]}")
                print(f"Election Ranking: {election_ranking} -> {new_election_ranking}")
                print(f"Election Scores: {votes} -> {new_votes}")
                print(f"Original Happiness:  {happinesses}")
                print(f"Strategic Happiness: {new_happinesses}")
                change_in_happiness = np.round((new_happinesses - happinesses) * 100/ np.maximum(new_happinesses, happinesses)).astype(int)
                print(f"Change In Happiness: {change_in_happiness} %")
                print()

        return



def generate_random_preferences_matrix(num_alternatives, num_voters):
    return np.array([np.random.permutation(num_alternatives) for _ in range(num_voters)]).T

random_matrix = generate_random_preferences_matrix(5, 8)

atva4_instance = ATVA4(random_matrix, 'borda')
strategic_scenarios = atva4_instance.run_strategic_btva_election()

