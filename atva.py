import numpy as np
import random
import itertools
import copy
import happiness as hpns
import strategic_voting_risk as svr

class BTVA:
    def __init__(self, voting_scheme, preference_matrix):
        self.voting_scheme = voting_scheme
        self.preference_matrix = preference_matrix
        self.num_alternatives, self.num_voters = preference_matrix.shape
        self.risk_of_strategic_voting = 0
        self.happinesses = np.zeros(self.num_voters)
        self.svr_scheme = 'count_strategic_votes'

    def run_non_strategic_election(self):
        scores = np.zeros(self.num_alternatives)
        if self.voting_scheme == 'plurality':
            for voter in range(self.num_voters):
                top_choice = self.preference_matrix[0, voter]
                scores[top_choice] += 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.exponential_decay_happiness(self.preference_matrix[:, voter], election_ranking)

        elif self.voting_scheme == 'voting_for_two':
            for voter in range(self.num_voters):
                top_choices = self.preference_matrix[:2, voter]
                for choice in top_choices:
                    scores[choice] += 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.k_binary_happiness(2, self.preference_matrix[:, voter], election_ranking)

        elif self.voting_scheme == 'anti_plurality':
            scores.fill(self.num_voters)
            for voter in range(self.num_voters):
                last_choice = self.preference_matrix[-1, voter]
                scores[last_choice] -= 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.binary_happiness(self.preference_matrix[:, voter], election_ranking, anti_plurality=True)

        elif self.voting_scheme == 'borda':
            for voter in range(self.num_voters):
                for rank, choice in enumerate(self.preference_matrix[:, voter]):
                    scores[choice] += (self.num_alternatives - rank - 1)
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.exp_decay_borda_style_happiness(
                    self.preference_matrix[:, voter], 
                    election_ranking,
                    polarization={'win_fr': 0.2, 'lose_fr': 0.2, 'wl_importance': 2})
        
        election_result = np.vstack((election_ranking, votes))
        return election_result

    def run_strategic_voting(self, election_result):
        election_ranking, votes = election_result 
        winner = election_ranking[0]
        incentives = np.zeros(self.num_voters)
        if self.voting_scheme == 'plurality':
            for voter in range(self.num_voters):
                voter_pref = self.preference_matrix[:, voter]
                voter_rank_for_winner = np.where(voter_pref == winner)[0][0]
                for contender in np.delete(voter_pref, voter_rank_for_winner):
                    voter_rank_for_contender = np.where(voter_pref == contender)[0][0]
                    if voter_rank_for_winner < voter_rank_for_contender:
                        continue
                    else:
                        other_alternatives = np.delete(voter_pref, [voter_rank_for_winner, voter_rank_for_contender])
                        strategic_preference = np.concatenate(([contender], other_alternatives, [winner]))
                        new_pref_matrix = np.copy(self.preference_matrix)
                        new_pref_matrix[:, voter] = strategic_preference
                        btva_strategic = BTVA(self.voting_scheme, new_pref_matrix)
                        new_election_ranking, _ = btva_strategic.run_non_strategic_election()
                        new_winner = new_election_ranking[0]
                        if new_winner != winner:
                            print(f'Voter {voter} strategic voting >>>')
                            print('New Preference Matrix')
                            print(new_pref_matrix)
                            print('New Winner:', new_winner)
                            incentives[voter] = 1 
            self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, incentives)
        return incentives

    def get_ordering_permutation(self):
        return self.preference_matrix.shape[0]

###############################################################################
# 1. Advanced Voting System: Addressing Voter Collusion (Multiple-Voter Manipulations)
###############################################################################
class BTVA_Collusion(BTVA):
    def run_collusive_strategic_voting(self, max_group_size=2):
        """
        Check if groups of voters (up to max_group_size) can coordinate
        their tactical vote to change the election outcome.
        """
        original_result = self.run_non_strategic_election()
        original_winner = original_result[0, 0]
        collusion_incentives = np.zeros(self.num_voters)
        
        # Try groups of voters (collusion) of sizes 2 to max_group_size
        for group_size in range(2, max_group_size + 1):
            for group in itertools.combinations(range(self.num_voters), group_size):
                new_pref_matrix = np.copy(self.preference_matrix)
                # For simplicity, let the colluding group agree on a common contender.
                candidate_scores = {}
                for voter in group:
                    for candidate in self.preference_matrix[:, voter]:
                        if candidate != original_winner:
                            rank = np.where(self.preference_matrix[:, voter] == candidate)[0][0]
                            candidate_scores[candidate] = candidate_scores.get(candidate, 0) + rank
                if not candidate_scores:
                    continue
                collusive_candidate = min(candidate_scores, key=candidate_scores.get)
                
                # Each voter in the group places the collusive candidate at the top and pushes the original winner last.
                for voter in group:
                    voter_pref = self.preference_matrix[:, voter]
                    new_ballot = [collusive_candidate] + [c for c in voter_pref if c not in [collusive_candidate, original_winner]] + [original_winner]
                    new_pref_matrix[:, voter] = new_ballot
                collusive_btva = BTVA(self.voting_scheme, new_pref_matrix)
                new_result = collusive_btva.run_non_strategic_election()
                new_winner = new_result[0, 0]
                if new_winner != original_winner:
                    for voter in group:
                        collusion_incentives[voter] = 1
                    print(f"Collusion by voters {group} can change the winner from {original_winner} to {new_winner}")
        self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, collusion_incentives)
        return collusion_incentives

###############################################################################
# 2. Advanced Voting System: Incorporating Counter-Strategic Voting
###############################################################################
class BTVA_CounterStrategic(BTVA):
    def run_counter_strategic_voting(self, max_rounds=3):
        """
        Iteratively allow voters to adjust their ballots in response to
        observed strategic moves (counter-strategic voting).
        """
        current_pref_matrix = np.copy(self.preference_matrix)
        round_num = 0
        overall_incentives = np.zeros(self.num_voters)
        
        while round_num < max_rounds:
            print(f"Round {round_num+1} of counter-strategic voting:")
            btva_instance = BTVA(self.voting_scheme, current_pref_matrix)
            election_result = btva_instance.run_non_strategic_election()
            original_winner = election_result[0, 0]
            round_incentives = np.zeros(self.num_voters)
            
            # For each voter, try a simple counter-move (swap top two positions)
            for voter in range(self.num_voters):
                voter_pref = current_pref_matrix[:, voter]
                # Only consider counter-moves if the voter currently votes for the winner
                if voter_pref[0] != original_winner:
                    continue
                new_ballot = np.copy(voter_pref)
                new_ballot[0], new_ballot[1] = new_ballot[1], new_ballot[0]
                new_pref_matrix = np.copy(current_pref_matrix)
                new_pref_matrix[:, voter] = new_ballot
                test_btva = BTVA(self.voting_scheme, new_pref_matrix)
                new_result = test_btva.run_non_strategic_election()
                new_winner = new_result[0, 0]
                if new_winner != original_winner:
                    round_incentives[voter] = 1
                    current_pref_matrix[:, voter] = new_ballot
                    print(f"Voter {voter} counter-changed ballot to: {new_ballot} resulting in new winner {new_winner}")
            if np.sum(round_incentives) == 0:
                break
            overall_incentives += round_incentives
            round_num += 1
        self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, overall_incentives)
        return overall_incentives

###############################################################################
# 3. Advanced Voting System: Imperfect Information about Voter Preferences
###############################################################################
class BTVA_ImperfectInfo(BTVA):
    def __init__(self, voting_scheme, preference_matrix, noise_level=0.1):
        super().__init__(voting_scheme, preference_matrix)
        self.noise_level = noise_level
        # Generate a noisy version of preferences to simulate imperfect information.
        self.noisy_preference_matrix = self.generate_noisy_preferences()

    def generate_noisy_preferences(self):
        """
        Create a noisy version of the true preference matrix by randomly swapping
        some candidates with probability defined by noise_level.
        """
        noisy_matrix = np.copy(self.preference_matrix)
        num_alternatives, num_voters = noisy_matrix.shape
        for voter in range(num_voters):
            for i in range(num_alternatives):
                if random.random() < self.noise_level:
                    swap_index = random.randint(0, num_alternatives - 1)
                    noisy_matrix[i, voter], noisy_matrix[swap_index, voter] = noisy_matrix[swap_index, voter], noisy_matrix[i, voter]
        return noisy_matrix

    def run_strategic_voting(self, election_result):
        """
        Override strategic voting to use the noisy (imperfect) preferences.
        """
        election_ranking, votes = election_result 
        winner = election_ranking[0]
        incentives = np.zeros(self.num_voters)
        for voter in range(self.num_voters):
            voter_noisy_pref = self.noisy_preference_matrix[:, voter]
            voter_rank_for_winner = np.where(voter_noisy_pref == winner)[0][0]
            # Choose the best contender from the noisy ranking that is not the winner
            potential_contenders = [c for c in voter_noisy_pref if c != winner]
            if not potential_contenders:
                continue
            contender = potential_contenders[0]
            voter_rank_for_contender = np.where(voter_noisy_pref == contender)[0][0]
            if voter_rank_for_winner > voter_rank_for_contender:
                strategic_ballot = np.concatenate(([contender], np.delete(voter_noisy_pref, np.where(voter_noisy_pref == contender))))
                new_pref_matrix = np.copy(self.preference_matrix)
                new_pref_matrix[:, voter] = strategic_ballot
                test_btva = BTVA(self.voting_scheme, new_pref_matrix)
                new_result = test_btva.run_non_strategic_election()
                new_winner = new_result[0, 0]
                if new_winner != winner:
                    incentives[voter] = 1
                    print(f"Voter {voter} (imperfect info) changed ballot to: {strategic_ballot} resulting in new winner {new_winner}")
        self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, incentives)
        return incentives

###############################################################################
# 4. Advanced Voting System: Concurrent Tactical Voting (Multiple Tactical Moves)
###############################################################################
class BTVA_ConcurrentTactical(BTVA):
    def run_concurrent_tactical_voting(self):
        """
        Allow each voter to try multiple tactical ballots concurrently and adopt
        the one that most improves their personal outcome.
        """
        original_result = self.run_non_strategic_election()
        original_winner = original_result[0, 0]
        best_incentives = np.zeros(self.num_voters)
        best_happiness = self.happinesses.copy()
        new_pref_matrix = np.copy(self.preference_matrix)
        
        # For each voter, try different reorderings of the top three positions.
        for voter in range(self.num_voters):
            voter_pref = self.preference_matrix[:, voter]
            best_ballot = voter_pref.copy()
            # Generate all permutations of the first three candidates.
            for perm in itertools.permutations(voter_pref[:3]):
                candidate_ballot = np.array(list(perm) + list(voter_pref[3:]))
                test_matrix = np.copy(new_pref_matrix)
                test_matrix[:, voter] = candidate_ballot
                test_btva = BTVA(self.voting_scheme, test_matrix)
                test_result = test_btva.run_non_strategic_election()
                new_winner = test_result[0, 0]
                # Calculate new happiness (using an example happiness function).
                new_happiness = hpns.exponential_decay_happiness(voter_pref, test_result[0])
                if new_happiness > best_happiness[voter] and new_winner != original_winner:
                    best_happiness[voter] = new_happiness
                    best_ballot = candidate_ballot
                    best_incentives[voter] = 1
            new_pref_matrix[:, voter] = best_ballot
        
        # Re-run the election with all updated ballots.
        final_btva = BTVA(self.voting_scheme, new_pref_matrix)
        final_result = final_btva.run_non_strategic_election()
        final_winner = final_result[0, 0]
        print("Final Winner after concurrent tactical voting:", final_winner)
        self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, best_incentives)
        return best_incentives

###############################################################################
# Example Usage
###############################################################################
pref_matrix = np.array([
    [0, 1, 3, 0, 2, 3, 2, 3],
    [2, 0, 0, 1, 4, 4, 4, 2],
    [1, 3, 2, 2, 1, 0, 0, 1],
    [3, 4, 4, 3, 3, 1, 1, 4],
    [4, 2, 1, 4, 0, 2, 3, 0]
])

print("=== Collusive Strategic Voting ===")
btva_collusion = BTVA_Collusion('plurality', pref_matrix)
collusive_incentives = btva_collusion.run_collusive_strategic_voting(max_group_size=2)
print("Collusive Incentives:", collusive_incentives)

print("\n=== Counter-Strategic Voting ===")
btva_counter = BTVA_CounterStrategic('plurality', pref_matrix)
counter_incentives = btva_counter.run_counter_strategic_voting(max_rounds=3)
print("Counter-Strategic Incentives:", counter_incentives)

print("\n=== Imperfect Information Strategic Voting ===")
btva_imperfect = BTVA_ImperfectInfo('plurality', pref_matrix, noise_level=0.2)
basic_result = btva_imperfect.run_non_strategic_election()
imperfect_incentives = btva_imperfect.run_strategic_voting(basic_result)
print("Imperfect Information Incentives:", imperfect_incentives)

print("\n=== Concurrent Tactical Voting ===")
btva_concurrent = BTVA_ConcurrentTactical('plurality', pref_matrix)
concurrent_incentives = btva_concurrent.run_concurrent_tactical_voting()
print("Concurrent Tactical Incentives:", concurrent_incentives)
