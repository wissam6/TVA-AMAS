import numpy as np
import random
import itertools
import copy
import happiness as hpns
import strategic_voting_risk as svr

class BTVA:
    def __init__(self, voting_scheme, preference_matrix, original_preference_matrix):
        self.voting_scheme = voting_scheme
        self.preference_matrix = preference_matrix
        self.num_alternatives, self.num_voters = preference_matrix.shape
        self.risk_of_strategic_voting = 0
        self.happinesses = np.zeros(self.num_voters)
        self.svr_scheme = 'count_strategic_votes'
        self.original_preference_matrix = original_preference_matrix

    def run_non_strategic_election(self):
        scores = np.zeros(self.num_alternatives)
        if self.voting_scheme == 'plurality':
            for voter in range(self.num_voters):
                top_choice = self.preference_matrix[0, voter]
                scores[top_choice] += 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.exponential_decay_happiness(self.original_preference_matrix[:, voter], election_ranking)

        elif self.voting_scheme == 'voting_for_two':
            for voter in range(self.num_voters):
                top_choices = self.preference_matrix[:2, voter]
                for choice in top_choices:
                    scores[choice] += 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.k_binary_happiness(2, self.original_preference_matrix[:, voter], election_ranking)

        elif self.voting_scheme == 'anti_plurality':
            scores.fill(self.num_voters)
            for voter in range(self.num_voters):
                last_choice = self.preference_matrix[-1, voter]
                scores[last_choice] -= 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.binary_happiness(self.original_preference_matrix[:, voter], election_ranking, anti_plurality=True)

        elif self.voting_scheme == 'borda':
            for voter in range(self.num_voters):
                for rank, choice in enumerate(self.preference_matrix[:, voter]):
                    if not np.isin(choice, range(self.num_alternatives)):
                        continue
                    else:
                        scores[choice] += (self.num_alternatives - rank - 1)
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.exp_decay_borda_style_happiness(
                    self.original_preference_matrix[:, voter], 
                    election_ranking,
                    polarization={'win_fr': 0.2, 'lose_fr': 0.2, 'wl_importance': 2})
        
        election_result = np.vstack((election_ranking, votes))
        return election_result
    
    # def calc_happinesses(self, preference_matrix, election_ranking, happiness_function, **kwargs):
    #     voters = np.arange(self.num_voters)
    #     happinesses = np.zeros(self.num_voters)
    #     for voter in voters:
    #         happinesses[voter] = happiness_function(preference_matrix[:, voter], election_ranking)
    #     return happinesses

    # def run_strategic_voting(self, election_result):

    #     election_ranking, votes = election_result 
    #     winner = election_ranking[0]
    #     incentives = np.zeros(self.num_voters)

    #     if self.voting_scheme == 'plurality':

    #         for voter in range(self.num_voters):
    #             voter_pref = self.preference_matrix[:, voter]
    #             voter_rank_for_winner = np.where(voter_pref == winner)[0][0]

    #             for contender in np.delete(voter_pref, voter_rank_for_winner):
    #                 voter_rank_for_contender = np.where(voter_pref == contender)[0][0]

    #                 if voter_rank_for_winner < voter_rank_for_contender:
    #                     continue
    #                 else:
    #                     other_alternatives = np.delete(voter_pref, [voter_rank_for_winner, voter_rank_for_contender])
    #                     strategic_preference = np.concatenate(([contender], other_alternatives, [winner]))
    #                     new_pref_matrix = np.copy(self.preference_matrix)
    #                     new_pref_matrix[:, voter] = strategic_preference
    #                     btva_strategic = BTVA(self.voting_scheme, new_pref_matrix)
    #                     new_election_ranking, _ = btva_strategic.run_non_strategic_election()
    #                     new_winner = new_election_ranking[0]
    #                     if new_winner != winner:
    #                         print(f'Voter {voter} strategic voting >>>')
    #                         print('New Preference Matrix')
    #                         print(new_pref_matrix)
    #                         print('New Winner:', new_winner)
    #                         incentives[voter] = 1 
    #         self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, incentives)

    #     elif self.voting_scheme == 'borda':
    #         # election_ranking, votes = election_result 
    #         # incentives = np.zeros(self.num_voters)
    #         contenders = election_ranking[1:]
    #         strategies = ['bullet', 'compromise', 'bury'] # 
            
    #         for voter in range(self.num_voters):
    #             voter_preference = self.preference_matrix[:, voter]  
    #             happiness = self.non_strategic_happinesses[voter]
    #             bullet_happiness = -1
    #             compromise_happiness = -1
    #             bury_happiness = -1

    #             for strategy in strategies:

    #                 if strategy == 'bullet':              
    #                     # in borda any change can potentially increase happiness, so every alternative is a contender
    #                     for contender in contenders:
    #                         # 1st strategy: bullet voting
    #                         strategic_preference = np.full_like(voter_preference, -1)
    #                         strategic_preference[0] = contender
    #                         strategic_preference_matrix = np.ndarray.copy(self.preference_matrix)
    #                         strategic_preference_matrix[:, voter] = strategic_preference
    #                         btva_strategic = BTVA('borda', strategic_preference_matrix)
    #                         new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
                            
    #                         new_happinesses = self.calc_happinesses(self.preference_matrix, new_election_ranking, hpns.exp_decay_borda_style_happiness)

    #                         if new_happinesses[voter] > happiness and new_happinesses[voter] > bullet_happiness:
    #                             bullet_happiness = new_happinesses[voter]
                                
    #                             # print(f'Voter {voter} strategic voting with BULLET>>>')
    #                             # print('New Preference Matrix')
    #                             # print(strategic_preference_matrix)
    #                             # print('New Election Ranking', new_election_ranking)
    #                             # print('New Election Scores ', new_votes)
    #                             # print("Original Voters' Happiness", np.array2string(self.non_strategic_happinesses, formatter={'float_kind': lambda x: f"{x:.2f}"}))
    #                             # print("Updated Voters' Happiness ", np.array2string(new_happinesses, formatter={'float_kind': lambda x: f"{x:.2f}"}))
    #                             # print()
    #                             # bullet_happiness = new_happinesses[voter]
    #                             # print(f'Bullet Happiness of voter {voter}:  {bullet_happiness}')
    #                             # print(f'Bulleting  "{contender}"')

    #                             if incentives[voter] == 0:
    #                                 incentives[voter] = 1
    #                 #COMPROMISE
    #                 elif strategy == 'compromise':
                        
    #                     for contender in contenders:
    #                         # Skip if the contender is already the voter's first preference, since compromising it wouldn;t change anything
    #                         if contender == voter_preference[0]:
    #                             continue
                            
    #                         original_index = np.where(voter_preference == contender)[0][0]
    #                         strategic_preference = np.copy(voter_preference)
    #                         strategic_preference = np.delete(strategic_preference, original_index)
    #                         strategic_preference = np.insert(strategic_preference, 0, contender)
                            
    #                         strategic_preference_matrix = np.copy(self.preference_matrix)
    #                         strategic_preference_matrix[:, voter] = strategic_preference

    #                         ## THESE PRINT STATEMENTS ARE TO SEE IF COMPROMISING A CONTENDER IS WORKING PROPERLY
    #                         # print(f'Voter: {voter}')
    #                         # print(f'Contender: {contender}')
    #                         # print(f'Voters original first preference: {voter_preference[0]}')
    #                         # print(f'Original Preference Matrix \n{self.preference_matrix}')
    #                         # print(f'Strategic Preference Matrix \n{strategic_preference_matrix}')
    #                         # print()
                            
                            
    #                         btva_strategic = BTVA('borda', strategic_preference_matrix)
    #                         new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
    #                         new_happinesses = self.calc_happinesses(self.preference_matrix, new_election_ranking, hpns.exp_decay_borda_style_happiness)
                            
    #                         # Check if this strategic move increases the voter's happiness
    #                         if new_happinesses[voter] > happiness and new_happinesses[voter] > compromise_happiness:
                                
    #                             compromise_happiness = new_happinesses[voter]
    #                             # print(f'Compromise Happiness of voter {voter}:  {compromise_happiness}')
    #                             # print(f'Comprimising  "{contender}"')
                                
    #                             if incentives[voter] == 0:
    #                                 incentives[voter] = 1

    #                 #BURY
    #                 elif strategy == 'bury':
    #                     for contender in contenders:
    #                         # Skip if the contender is already the last preference, since burying it wouldn't change anything
    #                         if contender == voter_preference[-1]:
    #                             continue

    #                         # Find the index of the contender in the voter's original preference list
    #                         original_index = np.where(voter_preference == contender)[0][0]
                                                
    #                         strategic_preference = np.copy(voter_preference)
    #                         strategic_preference = np.delete(strategic_preference, original_index)
    #                         strategic_preference = np.append(strategic_preference, contender)

    #                         strategic_preference_matrix = np.copy(self.preference_matrix)
    #                         strategic_preference_matrix[:, voter] = strategic_preference

    #                         ## THESE PRINT STATEMENTS ARE TO SEE IF BURYING A CONTENDER IS WORKING PROPERLY
    #                         # print(f'Voter: {voter}')
    #                         # print(f'Contender: {contender}')
    #                         # print(f"Voter's last preference: {voter_preference[-1]}")
    #                         # print(f'Original Preference Matrix \n{self.preference_matrix}')
    #                         # print(f'Strategic Preference Matrix \n{strategic_preference_matrix}')
    #                         # print()
                                                
    #                         btva_strategic = BTVA('borda', strategic_preference_matrix)
    #                         new_election_ranking, new_votes = btva_strategic.run_non_strategic_election()
    #                         new_happinesses = self.calc_happinesses(self.preference_matrix, new_election_ranking, hpns.exp_decay_borda_style_happiness)
                                                
    #                         # Check if this strategic move increases the voter's happiness
    #                         if new_happinesses[voter] > happiness and new_happinesses[voter] > bury_happiness:
    #                             bury_happiness = new_happinesses[voter]
    #                             # print(f'Burying Happiness of voter {voter}:  {bury_happiness}')
    #                             # print(f'Burying candidate "{contender}"')
                                                    
    #                             # Flag the voter as having an incentive to vote strategically if not already marked
    #                             if incentives[voter] == 0:
    #                                 incentives[voter] = 1

    #             ## THESE PRINT STATEMENTS ARE TO SEE IF THE STRATEGIC VOTING INCREASED THE VOTERS HAPPINESS, -1 MEANS THAT IT DIDN'T INCREASE
    #             # print(f'Burying Happiness of Voter {voter} : {bury_happiness}' )
    #             # print(f'Compromise Happiness of Voter {voter} : {compromise_happiness}' )
    #             # print(f'Bullet Happiness of Voter {voter} : {bullet_happiness}' )
    #             # print()

    #         self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, incentives)

    #     return incentives

    # def get_ordering_permutation(self):
    #     return self.preference_matrix.shape[0]

# ###############################################################################
# # 1. Advanced Voting System: Addressing Voter Collusion (Multiple-Voter Manipulations)
# ###############################################################################
# class BTVA_Collusion(BTVA):
#     def run_collusive_strategic_voting(self, max_group_size=2):
#         """
#         Check if groups of voters (up to max_group_size) can coordinate
#         their tactical vote to change the election outcome.
#         """
#         original_result = self.run_non_strategic_election()
#         original_winner = original_result[0, 0]
#         collusion_incentives = np.zeros(self.num_voters)
        
#         # Try groups of voters (collusion) of sizes 2 to max_group_size
#         for group_size in range(2, max_group_size + 1):
#             for group in itertools.combinations(range(self.num_voters), group_size):
#                 new_pref_matrix = np.copy(self.preference_matrix)
#                 # For simplicity, let the colluding group agree on a common contender.
#                 candidate_scores = {}
#                 for voter in group:
#                     for candidate in self.preference_matrix[:, voter]:
#                         if candidate != original_winner:
#                             rank = np.where(self.preference_matrix[:, voter] == candidate)[0][0]
#                             candidate_scores[candidate] = candidate_scores.get(candidate, 0) + rank
#                 if not candidate_scores:
#                     continue
#                 collusive_candidate = min(candidate_scores, key=candidate_scores.get)
                
#                 # Each voter in the group places the collusive candidate at the top and pushes the original winner last.
#                 for voter in group:
#                     voter_pref = self.preference_matrix[:, voter]
#                     new_ballot = [collusive_candidate] + [c for c in voter_pref if c not in [collusive_candidate, original_winner]] + [original_winner]
#                     new_pref_matrix[:, voter] = new_ballot
#                 collusive_btva = BTVA(self.voting_scheme, new_pref_matrix)
#                 new_result = collusive_btva.run_non_strategic_election()
#                 new_winner = new_result[0, 0]
#                 if new_winner != original_winner:
#                     for voter in group:
#                         collusion_incentives[voter] = 1
#                     print(f"Collusion by voters {group} can change the winner from {original_winner} to {new_winner}")
#         self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, collusion_incentives)
#         return collusion_incentives

###############################################################################
# 2. Advanced Voting System: Incorporating Counter-Strategic Voting
###############################################################################
class BTVA_CounterStrategic(BTVA):
    def run_counter_strategic_voting(self, max_rounds=3):
        """
        Iteratively allow voters to adjust their ballots in response to
        observed strategic moves (counter-strategic voting).
        """
        original_pref_matrix = np.copy(self.preference_matrix)
        current_pref_matrix = np.copy(self.preference_matrix)
        round_num = 0
        overall_incentives = np.zeros(self.num_voters)

        if(self.voting_scheme == 'plurality'):
            while round_num < max_rounds:
                print(f"Round {round_num+1} of counter-strategic voting:")
                btva_instance = BTVA(self.voting_scheme, current_pref_matrix, original_pref_matrix)
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

        elif self.voting_scheme == 'borda':

            round_incentives = np.zeros(self.num_voters)
            # round_incentives is defined outside the round loop so it persists across rounds
            # (we assume overall incentives are handled elsewhere)
            # current_pref_matrix is updated in-place when a voter votes strategically.
            while round_num < max_rounds:
                print(f"Round {round_num+1} of counter-strategic voting:")
                btva_instance = BTVA(self.voting_scheme, current_pref_matrix, original_pref_matrix)
                election_result = btva_instance.run_non_strategic_election()
                print("pref matrix IN THE BEGGINNING:\n",  current_pref_matrix)
                print("election result IN THE BEGGINNING:\n", election_result)
                election_ranking, _ = election_result
                print("hapinesses in the begginning of the round:", btva_instance.happinesses)
                current_winner = election_ranking[0]
               
                # Compute baseline happiness for each voter using the original (true) preference matrix
                btva_temp = BTVA(self.voting_scheme, self.preference_matrix, original_pref_matrix)
                non_strategic_result = btva_temp.run_non_strategic_election()
                baseline_happinesses = btva_temp.happinesses
                
                round_changes = False  # flag to check if any voter strategically votes in this round
                # Get the list of contenders (all candidates except the current winner)
                contenders = election_ranking[1:]
                
                # Iterate over voters in order
                for voter in range(self.num_voters):
                    voter_current_pref = current_pref_matrix[:, voter]
                    # Eligibility: skip if voter's first candidate is the winner or they've already voted strategically in a previous round.
                    if voter_current_pref[0] == current_winner or round_incentives[voter] == 1:
                        continue
                    
                    # Evaluate all three strategies to find the best move
                    best_happiness = baseline_happinesses[voter]
                    best_ballot = np.copy(voter_current_pref)
                    
                    # --- Bullet Strategy ---
                    # In Borda, bullet voting means setting all positions to -1 except the chosen contender in first place.
                    for contender in contenders:
                        strategic_preference = np.full_like(voter_current_pref, -1)
                        strategic_preference[0] = contender
                        strategic_pref_matrix = np.copy(current_pref_matrix)
                        strategic_pref_matrix[:, voter] = strategic_preference
                        btva_strategic = BTVA('borda', strategic_pref_matrix, original_pref_matrix)
                        new_result = btva_strategic.run_non_strategic_election()
                        new_election_ranking, _ = new_result
                        new_happinesses = btva_strategic.happinesses
                       
                        if new_happinesses[voter] > best_happiness:
                            best_happiness = new_happinesses[voter]
                            best_ballot = strategic_preference

                    # --- Compromise Strategy ---
                    for contender in contenders:
                        # Skip if contender is already at the top
                        if voter_current_pref[0] == contender:
                            continue
                        original_index = np.where(voter_current_pref == contender)[0][0]
                        strategic_preference = np.copy(voter_current_pref)
                        strategic_preference = np.delete(strategic_preference, original_index)
                        strategic_preference = np.insert(strategic_preference, 0, contender)
                        strategic_pref_matrix = np.copy(current_pref_matrix)
                        strategic_pref_matrix[:, voter] = strategic_preference
                        btva_strategic = BTVA('borda', strategic_pref_matrix, original_pref_matrix)
                        new_result = btva_strategic.run_non_strategic_election()
                        new_election_ranking, _ = new_result
                        new_happinesses = btva_strategic.happinesses
                        if new_happinesses[voter] > best_happiness:
                            best_happiness = new_happinesses[voter]
                            best_ballot = strategic_preference

                    # --- Bury Strategy ---
                    for contender in contenders:
                        # Skip if contender is already the last preference
                        if voter_current_pref[-1] == contender:
                            continue
                        original_index = np.where(voter_current_pref == contender)[0][0]
                        strategic_preference = np.copy(voter_current_pref)
                        strategic_preference = np.delete(strategic_preference, original_index)
                        strategic_preference = np.append(strategic_preference, contender)
                        strategic_pref_matrix = np.copy(current_pref_matrix)
                        strategic_pref_matrix[:, voter] = strategic_preference
                        btva_strategic = BTVA('borda', strategic_pref_matrix, original_pref_matrix)
                        new_result = btva_strategic.run_non_strategic_election()
                        new_election_ranking, _ = new_result
                        new_happinesses = btva_strategic.happinesses
                        if new_happinesses[voter] > best_happiness:
                            best_happiness = new_happinesses[voter]
                            best_ballot = strategic_preference
                
                    # If a better ballot was found, update the current preference matrix and mark this voter
                    if not np.array_equal(best_ballot, voter_current_pref):
                        # updated_matrix = np.copy(current_pref_matrix)
                        # updated_matrix[:, voter] = best_ballot
                        current_pref_matrix[:, voter] = best_ballot
                        round_incentives[voter] = 1
                        round_changes = True
                        # After the move, update the election result to reflect the change
                        btva_updated = BTVA(self.voting_scheme, current_pref_matrix, original_pref_matrix)
                        updated_result = btva_updated.run_non_strategic_election()
                        updated_ranking, _ = updated_result
                        new_winner = updated_ranking[0]
                        print(f"Voter {voter} strategic vote changed ballot to: {best_ballot}, new winner: {new_winner}")
                        print("END OF THE ROUND happinesses:", btva_updated.happinesses)
                        # Increase the round counter when someone strategically votes
                        round_num += 1
                        # Break out of the voter loop to start a new round immediately
                        break

                # End of round: print the current matrix and the new winner
                btva_round = BTVA(self.voting_scheme, current_pref_matrix, original_pref_matrix)
                round_result = btva_round.run_non_strategic_election()
                round_ranking, _ = round_result
                print("Preference matrix after this round:")
                print(current_pref_matrix)
                print("New result END OF THE ROUND:", round_ranking)
                print("New winner after this round:", round_ranking[0])
                
                # If no voter voted strategically in this full pass, exit the loop
                if not round_changes:
                    print("No voters changed their ballots in this round.")
                    break
            overall_incentives += round_incentives

        self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, overall_incentives)
        return overall_incentives

###############################################################################
# 3. Advanced Voting System: Imperfect Information about Voter Preferences
###############################################################################
# class BTVA_ImperfectInfo(BTVA):
#     def __init__(self, voting_scheme, preference_matrix, noise_level=0.1):
#         super().__init__(voting_scheme, preference_matrix)
#         self.noise_level = noise_level
#         # Generate a noisy version of preferences to simulate imperfect information.
#         self.noisy_preference_matrix = self.generate_noisy_preferences()

#     def generate_noisy_preferences(self):
#         """
#         Create a noisy version of the true preference matrix by randomly swapping
#         some candidates with probability defined by noise_level.
#         """
#         noisy_matrix = np.copy(self.preference_matrix)
#         num_alternatives, num_voters = noisy_matrix.shape
#         for voter in range(num_voters):
#             for i in range(num_alternatives):
#                 if random.random() < self.noise_level:
#                     swap_index = random.randint(0, num_alternatives - 1)
#                     noisy_matrix[i, voter], noisy_matrix[swap_index, voter] = noisy_matrix[swap_index, voter], noisy_matrix[i, voter]
#         return noisy_matrix

#     def run_strategic_voting(self, election_result):
#         """
#         Override strategic voting to use the noisy (imperfect) preferences.
#         """
#         election_ranking, votes = election_result 
#         winner = election_ranking[0]
#         incentives = np.zeros(self.num_voters)
#         for voter in range(self.num_voters):
#             voter_noisy_pref = self.noisy_preference_matrix[:, voter]
#             voter_rank_for_winner = np.where(voter_noisy_pref == winner)[0][0]
#             # Choose the best contender from the noisy ranking that is not the winner
#             potential_contenders = [c for c in voter_noisy_pref if c != winner]
#             if not potential_contenders:
#                 continue
#             contender = potential_contenders[0]
#             voter_rank_for_contender = np.where(voter_noisy_pref == contender)[0][0]
#             if voter_rank_for_winner > voter_rank_for_contender:
#                 strategic_ballot = np.concatenate(([contender], np.delete(voter_noisy_pref, np.where(voter_noisy_pref == contender))))
#                 new_pref_matrix = np.copy(self.preference_matrix)
#                 new_pref_matrix[:, voter] = strategic_ballot
#                 test_btva = BTVA(self.voting_scheme, new_pref_matrix)
#                 new_result = test_btva.run_non_strategic_election()
#                 new_winner = new_result[0, 0]
#                 if new_winner != winner:
#                     incentives[voter] = 1
#                     print(f"Voter {voter} (imperfect info) changed ballot to: {strategic_ballot} resulting in new winner {new_winner}")
#         self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, incentives)
#         return incentives

# ###############################################################################
# # 4. Advanced Voting System: Concurrent Tactical Voting (Multiple Tactical Moves)
# ###############################################################################
# class BTVA_ConcurrentTactical(BTVA):
#     def run_concurrent_tactical_voting(self):
#         """
#         Allow each voter to try multiple tactical ballots concurrently and adopt
#         the one that most improves their personal outcome.
#         """
#         original_result = self.run_non_strategic_election()
#         original_winner = original_result[0, 0]
#         best_incentives = np.zeros(self.num_voters)
#         best_happiness = self.happinesses.copy()
#         new_pref_matrix = np.copy(self.preference_matrix)
        
#         # For each voter, try different reorderings of the top three positions.
#         for voter in range(self.num_voters):
#             voter_pref = self.preference_matrix[:, voter]
#             best_ballot = voter_pref.copy()
#             # Generate all permutations of the first three candidates.
#             for perm in itertools.permutations(voter_pref[:3]):
#                 candidate_ballot = np.array(list(perm) + list(voter_pref[3:]))
#                 test_matrix = np.copy(new_pref_matrix)
#                 test_matrix[:, voter] = candidate_ballot
#                 test_btva = BTVA(self.voting_scheme, test_matrix)
#                 test_result = test_btva.run_non_strategic_election()
#                 new_winner = test_result[0, 0]
#                 # Calculate new happiness (using an example happiness function).
#                 new_happiness = hpns.exponential_decay_happiness(voter_pref, test_result[0])
#                 if new_happiness > best_happiness[voter] and new_winner != original_winner:
#                     best_happiness[voter] = new_happiness
#                     best_ballot = candidate_ballot
#                     best_incentives[voter] = 1
#             new_pref_matrix[:, voter] = best_ballot
        
#         # Re-run the election with all updated ballots.
#         final_btva = BTVA(self.voting_scheme, new_pref_matrix)
#         final_result = final_btva.run_non_strategic_election()
#         final_winner = final_result[0, 0]
#         print("Final Winner after concurrent tactical voting:", final_winner)
#         self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, best_incentives)
#         return best_incentives

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

# print("=== Collusive Strategic Voting ===")
# btva_collusion = BTVA_Collusion('plurality', pref_matrix)
# collusive_incentives = btva_collusion.run_collusive_strategic_voting(max_group_size=2)
# print("Collusive Incentives:", collusive_incentives)

print("\n=== Counter-Strategic Voting ===")
btva_counter = BTVA_CounterStrategic('borda', pref_matrix, pref_matrix)
counter_incentives = btva_counter.run_counter_strategic_voting(max_rounds=3)
print("Counter-Strategic Incentives:", counter_incentives)

# print("\n=== Imperfect Information Strategic Voting ===")
# btva_imperfect = BTVA_ImperfectInfo('plurality', pref_matrix, noise_level=0.2)
# basic_result = btva_imperfect.run_non_strategic_election()
# imperfect_incentives = btva_imperfect.run_strategic_voting(basic_result)
# print("Imperfect Information Incentives:", imperfect_incentives)

# print("\n=== Concurrent Tactical Voting ===")
# btva_concurrent = BTVA_ConcurrentTactical('plurality', pref_matrix)
# concurrent_incentives = btva_concurrent.run_concurrent_tactical_voting()
# print("Concurrent Tactical Incentives:", concurrent_incentives)