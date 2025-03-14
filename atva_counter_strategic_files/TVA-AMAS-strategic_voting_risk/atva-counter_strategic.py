import numpy as np
import random
import itertools
import copy
import happiness as hpns
import strategic_voting_risk as svr
import matplotlib.pyplot as plt
import csv
import os

class BTVA:
    def __init__(self, voting_scheme, preference_matrix, original_preference_matrix):
        self.voting_scheme = voting_scheme
        self.preference_matrix = preference_matrix
        self.num_alternatives, self.num_voters = preference_matrix.shape
        self.risk_of_strategic_voting = 0
        if self.voting_scheme == 'borda':
            self.happinesses = np.zeros(self.num_voters)
        elif self.voting_scheme == 'plurality':
            self.happinesses = np.zeros(self.num_voters)
        elif self.voting_scheme == 'voting_for_two':    
            self.happinesses = np.zeros(self.num_voters)
        elif self.voting_scheme == 'anti_plurality':
            self.happinesses = np.zeros(self.num_voters)
        else:
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
                    polarization={'win_fr': 1, 'lose_fr': 0, 'wl_importance': 2})
        
        election_result = np.vstack((election_ranking, votes))
        return election_result

###############################################################################
# 2. Advanced Voting System: Incorporating Counter-Strategic Voting
###############################################################################
class BTVA_CounterStrategic(BTVA):
    def run_counter_strategic_voting(self, max_rounds):
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
                    test_btva = BTVA(self.voting_scheme, new_pref_matrix, original_pref_matrix)
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
                #print(f"ROUND {round_num+1} OF COUNTER-STRATEGIC VOTING:------------------------------------------------------")
                btva_instance = BTVA(self.voting_scheme, current_pref_matrix, original_pref_matrix)
                election_result = btva_instance.run_non_strategic_election()
                #print("pref matrix IN THE BEGGINNING:\n",  current_pref_matrix)
                #print("election result IN THE BEGGINNING:\n", election_result)
                election_ranking, _ = election_result
                #print("hapinesses in the begginning of the round:", btva_instance.happinesses)
                current_winner = election_ranking[0]
                #print("\n")
               
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
                        print("Overall happinesses:", np.sum(btva_updated.happinesses) / btva_updated.num_voters)
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
                print("\n")
                self.happinesses = btva_round.happinesses.copy()
                
                # If no voter voted strategically in this full pass, exit the loop
                if not round_changes:
                    print("No voters changed their ballots in this round.")
                    print("\n")
                    break
            overall_incentives += round_incentives

        
        self.risk_of_strategic_voting = svr.get_strategic_voting_risk(self, overall_incentives)

        strategic_voting_risk = self.risk_of_strategic_voting
        overall_happiness = np.sum(self.happinesses) / self.num_voters

        # results_file = "results2.csv"

        # # Check if file exists; if not, write the header
        # file_exists = os.path.isfile(results_file)
        # with open(results_file, "a", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     if not file_exists:
        #         writer.writerow(["max_rounds", "strategic_voting_risk", "overall_happiness"])
            
        #     # Write the results for this run
        #     writer.writerow([max_rounds, strategic_voting_risk, overall_happiness])
        
        # results_file = "results3.csv"

        # # Check if file exists; if not, write the header
        # file_exists = os.path.isfile(results_file)
        # with open(results_file, "a", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     if not file_exists:
        #         writer.writerow(["num_voters", "strategic_voting_risk", "overall_happiness"])
            
        #     # Write the results for this run
        #     writer.writerow([self.num_voters, strategic_voting_risk, overall_happiness])

        # results_file = "results4.csv"

        # # Check if file exists; if not, write the header
        # file_exists = os.path.isfile(results_file)
        # with open(results_file, "a", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     if not file_exists:
        #         writer.writerow(["num_alternatives", "strategic_voting_risk", "overall_happiness"])
            
        #     # Write the results for this run
        #     writer.writerow([self.num_alternatives, strategic_voting_risk, overall_happiness])

        
        return overall_incentives


## FUNCTION TO GENERATE RANDOM PREFERENCE MATRICES   
def generate_random_preferences(voting_scheme, num_voters, num_alternatives):
        if voting_scheme == "borda":
            preferences = np.array([np.random.permutation(num_alternatives) for _ in range(num_voters)]).T
            return preferences

        elif voting_scheme == "plurality":
            # Each voter votes for one candidate: one 1 and the rest 0's.
            preferences = np.zeros((num_alternatives, num_voters), dtype=int)
            for voter in range(num_voters):
                choice = np.random.randint(0, num_alternatives)
                preferences[choice, voter] = 1
            return preferences

        elif voting_scheme in ["veto", "anti-plurality"]:
            # Each voter approves all candidates except one (their veto).
            preferences = np.ones((num_alternatives, num_voters), dtype=int)
            for voter in range(num_voters):
                veto_choice = np.random.randint(0, num_alternatives)
                preferences[veto_choice, voter] = 0
            return preferences

        elif voting_scheme == "voting_for_two":
            # Each voter votes for two candidates: two 1's and the rest 0's.
            preferences = np.zeros((num_alternatives, num_voters), dtype=int)
            for voter in range(num_voters):
                choices = np.random.choice(num_alternatives, size=2, replace=False)
                preferences[choices, voter] = 1
            return preferences
        else:
            print("Invalid voting scheme")
            return None

pref_matrix = np.array([
    [0, 1, 3, 0, 2, 3, 2, 3],
    [2, 0, 0, 1, 4, 4, 4, 2],
    [1, 3, 2, 2, 1, 0, 0, 1],
    [3, 4, 4, 3, 3, 1, 1, 4],
    [4, 2, 1, 4, 0, 2, 3, 0]
])

pref_matrix2 = np.array([
    [4, 1, 3, 4, 4, 3, 2, 2],
    [1, 2, 1, 3, 0, 2, 4, 4],
    [2, 4, 0, 2, 1, 0, 3, 1],
    [0, 0, 2, 0, 3, 4, 0, 0],
    [3, 3, 4, 1, 2, 1, 1, 3]
 ])

pref_matrix3 = np.array([
    [3, 4, 2, 4, 2, 4, 2, 2],
    [0, 2, 4, 2, 0, 0, 0, 3],
    [1, 1, 0, 0, 1, 3, 4, 0],
    [2, 0, 1, 1, 3, 1, 1, 1],
    [4, 3, 3, 3, 4, 2, 3, 4]
 ])
np.set_printoptions(linewidth=200, threshold=np.inf)

num_voters = 5
#num_alternatives = 30
max_rounds = 3


max_rounds_list = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50]
num_voters_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150]
# num_voters_list = []
num_alternatives_list =  [6] #[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150]

for num_alternatives in num_alternatives_list:
    
    matrix = generate_random_preferences("borda",num_voters , 7)
    
    # if max_rounds == 0:
    #     print("\n=== Non-Strategic Voting ===")
    #     btva = BTVA('borda', matrix, matrix)
    #     election_result = btva.run_non_strategic_election()
    #     election_ranking, _ = election_result
    #     winner = election_ranking[0]
    #     print("Winner:", winner)
    #     print("Happiness:", np.sum(btva.happinesses) / num_voters)
    #     print("Risk of Strategic Voting:", btva.risk_of_strategic_voting)
    
    btva_counter = BTVA_CounterStrategic('plurality', matrix, matrix)
    counter_incentives = btva_counter.run_counter_strategic_voting(max_rounds)
