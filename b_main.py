import numpy as np
from btva import BTVA
from b_plurality import BPlurality
from b_anti_plurality import BAntiPlurality
from b_voting_for_two import BVotingForTwo
from b_borda import BBorda
from happiness import *
from risk import *
from functools import partial

def generate_random_preferences_matrix(num_alternatives, num_voters):
    return np.array([np.random.permutation(num_alternatives) for _ in range(num_voters)]).T

btva_classes_dict = {
    'plurality': BPlurality,
    'anti_plurality': BAntiPlurality,
    'voting_for_two': BVotingForTwo,
    'borda': BBorda
}

num_alternatives = 5
num_voters = 8
btva_happiness_functions_dict = {
    'plurality': distance_sensitive_happiness,
    'anti_plurality': distance_sensitive_happiness,
    'voting_for_two': partial(
        exp_decay_borda_style_happiness,
        polarization={'win_fr': 2/num_alternatives, 'lose_fr': 0, 'wl_importance': 1}
    ),
    'borda': exp_decay_borda_style_happiness
}


random_matrix = generate_random_preferences_matrix(num_alternatives, num_voters)
voting_scheme = 'borda'
btva_instance = btva_classes_dict[voting_scheme](random_matrix, btva_happiness_functions_dict[voting_scheme])

# Running a non-strategic election
election_result = btva_instance.run_non_strategic_election()
election_ranking, votes = election_result

happinesses = btva_instance.calc_happinesses(election_ranking)
btva_instance.non_strategic_happinesses = happinesses

print("Preference Matrix:")
print(random_matrix)
print()
print(f"NON-STRATEGIC {voting_scheme.upper().replace('_', '-')} ELECTION >>>")
print("Winner:", election_ranking[0])
print("Election Ranking:", election_ranking)
print("Election Scores:", votes)
print("Voters' Happiness:", happinesses)
print()


# Running a strategic election
strategic_scenarios = btva_instance.run_strategic_election(election_result)
voter_strategic_gains = btva_instance.calc_strategic_gains(strategic_scenarios)
btva_instance.pretty_print_scenarios(strategic_scenarios)
print('VOTER STRATEGIC HAPPINESS GAINS:')
print(np.array2string(voter_strategic_gains * 100, formatter={'float_kind': lambda x: f"{x:.0f}%"}))
print(f'Risk of Strategic Voting: {gain_percentile_risk(voter_strategic_gains, percentile=75, only_consider_gainers=True):.2f}')