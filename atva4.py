import numpy as np
from btva import BTVA
from b_plurality import BPlurality
from b_anti_plurality import BAntiPlurality
from b_voting_for_two import BVotingForTwo
from b_borda import BBorda
from happiness import *
from risk import *
from helper_functions import *
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


    def run_potential_concurrent_strategic_elections(self):
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
        print(f"::: NON-STRATEGIC {self.voting_scheme.upper().replace('_', '-')} ELECTION :::")
        print("Non-strategic Voting Outcome (O):")
        print(f"Winner -> {election_ranking[0]}")
        print(f"Ranking -> {election_ranking}")
        print(f"Votes -> {votes}")
        print("Voters' Happiness Levels (Hi):", happinesses)
        print(f"Overall Voter Happiness Level (H): {np.around(sum(happinesses), 2)} (Out of max possible {btva_instance.num_voters})")
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
            print("=" * 37)
            print(f"POTENTIAL CONCURRENT STRATEGIC VOTING")
            print("=" * 37)
            print(f"Potential strategic voters are -> {', '.join(str(v) for v in strategic_voters)}")
            
        print()

        potential_change_in_happinesses = {voter: {'participated': 0, 'not_participated': 0} for voter in strategic_voters}
        num_combos = 0
        for n in range(2, len(strategic_voters) + 1):
            num_combos += 1
            print("-" * 40)
            print(f"{n} voters simultaneous strategical voting")
            print("-" * 40)
            print()
            for combo in itertools.combinations(strategic_voters, n):
                print(f":: voters {', '.join(str(v) for v in combo)} simultaneously voting strategically..")
                strategic_preference_matrix = np.copy(self.preference_matrix)
                combo = np.array(combo)
                new_pref_columns = np.column_stack(
                    [strategic_scenarios[voter]['strategic preference matrix'][:, voter] for voter in combo]
                )
                strategic_preference_matrix[:, combo] = new_pref_columns
                print_side_by_side(self.preference_matrix, strategic_preference_matrix)
                print()

                # Running a non-strategic btva election
                new_btva_instance = btva_classes_dict[self.voting_scheme](strategic_preference_matrix, self.btva_happiness_functions_dict[self.voting_scheme])
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

                for voter in strategic_voters:
                    if voter in combo:
                        potential_change_in_happinesses[voter]['participated'] += change_in_happiness[voter]
                    else:
                        potential_change_in_happinesses[voter]['not_participated'] += change_in_happiness[voter]

            for voter in strategic_voters:
                potential_change_in_happinesses[voter]['participated'] /= num_combos
                potential_change_in_happinesses[voter]['not_participated'] /= num_combos
            
        # Printing potential gains/losses in case of participating or not participating
        if len(strategic_voters) > 1:
            print(f"{'Voter':<10}{'Participated':<20}{'Not Participated':<20}")
            for voter, vals in potential_change_in_happinesses.items():
                print(f"{voter:<10}{vals['participated']:<20.2f}{vals['not_participated']:<20.2f}")

        return potential_change_in_happinesses, strategic_scenarios

    def run_final_concurrent_strategic_election(self, potential_change_in_happinesses, strategic_scenarios):
        btva_classes_dict = {
            'plurality': BPlurality,
            'anti_plurality': BAntiPlurality,
            'voting_for_two': BVotingForTwo,
            'borda': BBorda
        }

        potential_strategic_voters = [voter for voter, strategy in enumerate(strategic_scenarios) if strategy]
        strategic_voters = []
        for voter in potential_strategic_voters:
            if potential_change_in_happinesses[voter]['participated'] > potential_change_in_happinesses[voter]['not_participated']:
                strategic_voters.append(voter)
        
        print()
        print("=" * 34)
        print(f"ACTUAL CONCURRENT STRATEGIC VOTING")
        print("=" * 34)

        if len(strategic_voters) == 0:
            print("There are no strategic voters remaining in this election!")
        elif len(strategic_voters) == 1:
            print(f"The only strategic voter is {strategic_voters[0]}; No concurrent voting.")
        else:

            btva_instance = btva_classes_dict[self.voting_scheme](self.preference_matrix, self.btva_happiness_functions_dict[self.voting_scheme])
            election_result = btva_instance.run_non_strategic_election()
            election_ranking, votes = election_result
            happinesses = btva_instance.calc_happinesses(election_ranking)
            
            print(f"Final strategic voters are -> {', '.join(str(v) for v in strategic_voters)}")
            print()

            print(f":: voters {', '.join(str(v) for v in strategic_voters)} simultaneously voting strategically..")
            strategic_preference_matrix = np.copy(self.preference_matrix)
            strategic_voters = np.array(strategic_voters)
            new_pref_columns = np.column_stack(
                [strategic_scenarios[voter]['strategic preference matrix'][:, voter] for voter in strategic_voters]
            )
            strategic_preference_matrix[:, strategic_voters] = new_pref_columns
            
            print_side_by_side(self.preference_matrix, strategic_preference_matrix)
            print()

            # Running a non-strategic btva election
            new_btva_instance = btva_classes_dict[self.voting_scheme](strategic_preference_matrix, self.btva_happiness_functions_dict[self.voting_scheme])        
            new_election_result = new_btva_instance.run_non_strategic_election()
            new_election_ranking, new_votes = new_election_result
            new_happinesses = new_btva_instance.calc_happinesses(new_election_ranking, self.preference_matrix)
            new_btva_instance.non_strategic_happinesses = new_happinesses

            print("Change in Voting Outcome (O):")
            print(f"Winner: {election_ranking[0]} -> {new_election_ranking[0]}")
            print(f"Ranking: {election_ranking} -> {new_election_ranking}")
            print(f"Votes: {votes} -> {new_votes}")
            print(f"Original Happiness (Hi):  {happinesses}")
            print(f"Strategic Happiness (H~i): {new_happinesses}")
            change_in_happiness = np.round((new_happinesses - happinesses) * 100/ np.maximum(new_happinesses, happinesses)).astype(int)
            change_in_overall_happiness = np.round((sum(new_happinesses) - sum(happinesses)) * 100/ np.maximum(sum(new_happinesses), sum(happinesses))).astype(int)
            print(f"Overall Voter Happiness Level (H): {np.around(sum(happinesses), 2)} (Out of max possible {self.num_voters})")
            print(f"Strategic Voter Happiness Level (H~): {np.around(sum(new_happinesses), 2)} (Out of max possible {self.num_voters})")
            print(f"Change In Individual Happiness: {change_in_happiness} %")
            print(f"Change In Overall Happiness: {change_in_overall_happiness} %")
            print(f"Change In Happiness: {change_in_happiness} %")
            print()
            
            print(f'Overall Risk of Strategic Voting: {np.max(change_in_happiness[strategic_voters])} %')
        return


random_matrix = generate_random_preferences_matrix(5, 8)

atva4_instance = ATVA4(random_matrix, 'borda')
potential_change_in_happinesses, strategic_scenarios = atva4_instance.run_potential_concurrent_strategic_elections()
atva4_instance.run_final_concurrent_strategic_election(potential_change_in_happinesses, strategic_scenarios)

