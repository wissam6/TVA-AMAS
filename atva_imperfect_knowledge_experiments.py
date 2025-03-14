import numpy as np
import matplotlib.pyplot as plt

from atva_imperfect_knowledge import BTVA_ImperfectInfo
from b_main import generate_random_preferences_matrix

def run_single_ik_trial(
    voting_scheme, 
    num_alternatives, 
    num_voters, 
    noise_level=0.2, 
    strategic_voter_idx=0
):
    """
    - Generates a random full preference matrix using generate_random_preferences_matrix.
    - Constructs a BTVA_ImperfectInfo object with the given voting_scheme and noise_level.
    - Runs non-strategic election (optional) and then run_strategic_voting.
    - Returns a dict with final risk, the strategic voter's final happiness, 
      and possibly other measures (prob_of_regret, expected_regret).
    """

    full_matrix = generate_random_preferences_matrix(
        num_alternatives, 
        num_voters
    )

    btva_imperfect = BTVA_ImperfectInfo(
        voting_scheme=voting_scheme,
        original_pref_matrix=full_matrix,
        noise_level=noise_level,
        strategic_voter_idx=strategic_voter_idx
    )

    btva_imperfect.run_non_strategic_election()

    _ = btva_imperfect.run_strategic_voting(None)

    final_risk = btva_imperfect.risk_of_strategic_voting

    final_sincere_hap = getattr(btva_imperfect, 'avg_sincere_happiness', np.nan)
    final_chosen_hap  = getattr(btva_imperfect, 'avg_chosen_happiness', np.nan)

    return {
        'risk': final_risk,
        'sincere_happiness': final_sincere_hap,
        'final_happiness': final_chosen_hap
    }

def experiment_vary_num_voters(
    voting_schemes, 
    voter_counts, 
    num_alternatives=5,
    noise_level=0.2,
    trials_per_setting=10,
    strategic_voter_idx=0
):
    """
    For each voting scheme in 'voting_schemes' and each number of voters in 'voter_counts',
    run 'trials_per_setting' independent random trials, then average the results.
    
    Plots the average risk and average happiness in one figure, 
    showing a separate line for each voting scheme.
    """
    results = {}
    for scheme in voting_schemes:
        results[scheme] = {
            'voters': [],
            'avg_risk': [],
            'avg_sincere_happiness': [],
            'avg_final_hap' : []
        }


    for scheme in voting_schemes:
        print(f"\n=== Experiment: {scheme} while varying #Voters ===")
        for N in voter_counts:
            all_risks = []
            all_happiness = []
            sincere_happs = []

            for _ in range(trials_per_setting):
                outcome = run_single_ik_trial(
                    voting_scheme=scheme,
                    num_alternatives=num_alternatives,
                    num_voters=N,
                    noise_level=noise_level,
                    strategic_voter_idx=strategic_voter_idx
                )
                all_risks.append(outcome['risk'])
                sincere_happs.append(outcome['sincere_happiness'])
                all_happiness.append(outcome['final_happiness'])

            avg_risk = np.mean(all_risks)
            avg_sincere = np.mean(sincere_happs)
            avg_hap  = np.mean(all_happiness)

            results[scheme]['voters'].append(N)
            results[scheme]['avg_risk'].append(avg_risk)
            results[scheme]['avg_final_hap'].append(avg_hap)
            results[scheme]['avg_sincere_happiness'].append(avg_sincere)


    plt.figure(figsize=(8,6))
    for scheme in voting_schemes:
        x_vals = results[scheme]['voters']
        y_risk = results[scheme]['avg_risk']
        y_sincere = results[scheme]['avg_sincere_happiness']
        y_hap  = results[scheme]['avg_final_hap']

        plt.plot(x_vals, y_risk, marker='o', label=f'{scheme} Risk')

        plt.plot(x_vals, y_sincere, marker='x', label=f'{scheme} Sincere Happiness')
        plt.plot(x_vals, y_hap, marker='x', label=f'{scheme} Final Happiness')

    plt.title(f"Varying Number of Voters (alts={num_alternatives}, noise={noise_level})")
    plt.xlabel("Number of Voters")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()

    return results


def experiment_vary_num_alternatives(
    voting_schemes,
    alternatives_list,
    num_voters=20,
    noise_level=0.2,
    trials_per_setting=10,
    strategic_voter_idx=0
):
    results = {}
    for scheme in voting_schemes:
        results[scheme] = {
            'alts': [],
            'avg_risk': [],
            'avg_sincere_happiness': [],
            'avg_final_hap' : []
        }

    for scheme in voting_schemes:
        print(f"\n=== Experiment: {scheme} while varying #Alternatives ===")
        for M in alternatives_list:
            all_risks = []
            all_happiness = []
            sincere_happs = []

            for _ in range(trials_per_setting):
                outcome = run_single_ik_trial(
                    voting_scheme=scheme,
                    num_alternatives=M,
                    num_voters=num_voters,
                    noise_level=noise_level,
                    strategic_voter_idx=strategic_voter_idx
                )
                all_risks.append(outcome['risk'])
                sincere_happs.append(outcome['sincere_happiness'])
                all_happiness.append(outcome['final_happiness'])

            avg_risk = np.mean(all_risks)
            avg_sincere = np.mean(sincere_happs)
            avg_hap  = np.mean(all_happiness)

            results[scheme]['alts'].append(M)
            results[scheme]['avg_risk'].append(avg_risk)
            results[scheme]['avg_final_hap'].append(avg_hap)
            results[scheme]['avg_sincere_happiness'].append(avg_sincere)

    plt.figure(figsize=(8,6))
    for scheme in voting_schemes:
        x_vals = results[scheme]['alts']
        y_risk = results[scheme]['avg_risk']
        y_hap  = results[scheme]['avg_final_hap']
        y_sincere = results[scheme]['avg_sincere_happiness']

        plt.plot(x_vals, y_risk, marker='o', label=f'{scheme} Risk')
        plt.plot(x_vals, y_hap, marker='x', label=f'{scheme} Final Happiness')
        plt.plot(x_vals, y_sincere, marker='x', label=f'{scheme} Sincere Happiness')

    plt.title(f"Varying Number of Alternatives (voters={num_voters}, noise={noise_level})")
    plt.xlabel("Number of Alternatives")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()

    return results


def experiment_vary_noise_level(
    voting_schemes,
    noise_levels,
    num_voters=20,
    num_alternatives=5,
    trials_per_setting=10,
    strategic_voter_idx=0
):
    results = {}
    for scheme in voting_schemes:
        results[scheme] = {
            'noise': [],
            'avg_risk': [],
            'avg_sincere_happiness': [],
            'avg_final_hap' : []
        }

    for scheme in voting_schemes:
        print(f"\n=== Experiment: {scheme} while varying noise level ===")
        for nl in noise_levels:
            all_risks = []
            all_happiness = []
            sincere_happs = []

            for _ in range(trials_per_setting):
                outcome = run_single_ik_trial(
                    voting_scheme=scheme,
                    num_alternatives=num_alternatives,
                    num_voters=num_voters,
                    noise_level=nl,
                    strategic_voter_idx=strategic_voter_idx
                )
                all_risks.append(outcome['risk'])
                sincere_happs.append(outcome['sincere_happiness'])
                all_happiness.append(outcome['final_happiness'])


            avg_risk = np.mean(all_risks)
            avg_sincere = np.mean(sincere_happs)
            avg_hap  = np.mean(all_happiness)

            results[scheme]['noise'].append(nl)
            results[scheme]['avg_risk'].append(avg_risk)
            results[scheme]['avg_final_hap'].append(avg_hap)
            results[scheme]['avg_sincere_happiness'].append(avg_sincere)

    plt.figure(figsize=(8,6))
    for scheme in voting_schemes:
        x_vals = results[scheme]['noise']
        y_risk = results[scheme]['avg_risk']
        y_hap  = results[scheme]['avg_final_hap']
        y_sincere = results[scheme]['avg_sincere_happiness']

        
        plt.plot(x_vals, y_risk, marker='o', label=f'{scheme} Risk')
        plt.plot(x_vals, y_hap, marker='x', label=f'{scheme} Final Happiness')
        plt.plot(x_vals, y_sincere, marker='x', label=f'{scheme} Sincere Happiness')

    plt.title(f"Varying Noise Level (voters={num_voters}, alts={num_alternatives})")
    plt.xlabel("Noise Level")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()

    return results



if __name__ == "__main__":
    voting_schemes = ['borda', 'plurality']
    
    # A) Vary number of voters
    voter_range = [5, 10, 15, 20, 30]
    experiment_vary_num_voters(
        voting_schemes=voting_schemes,
        voter_counts=voter_range,
        num_alternatives=5,
        noise_level=0.2,
        trials_per_setting=5,
        strategic_voter_idx=0
    )
    
    # B) Vary number of alternatives
    alt_range = [5, 6 , 7, 8]
    experiment_vary_num_alternatives(
        voting_schemes=voting_schemes,
        alternatives_list=alt_range,
        num_voters=15,
        noise_level=0.2,
        trials_per_setting=5,
        strategic_voter_idx=0
    ) 
    
    # C) Vary noise level
    noise_list = [0.0, 0.1, 0.2, 0.3, 0.5]
    experiment_vary_noise_level(
        voting_schemes=voting_schemes,
        noise_levels=noise_list,
        num_voters=10,
        num_alternatives=5,
        trials_per_setting=5,
        strategic_voter_idx=0
    )
