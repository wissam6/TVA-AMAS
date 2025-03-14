import numpy as np
import itertools
import happiness as hpns
import risk as svr
from b_main import generate_random_preferences_matrix

class BTVA:
    def __init__(self, voting_scheme, preference_matrix, original_preference_matrix):
        """
        voting_scheme : str
            E.g. 'plurality' or 'borda'
        preference_matrix : np.ndarray
            The matrix actually used for the election (shape = (num_alternatives, num_voters)).
        original_preference_matrix : np.ndarray
            The fully known "true" preference matrix (for computing happiness).
        """
        self.voting_scheme = voting_scheme
        self.preference_matrix = preference_matrix
        self.num_alternatives, self.num_voters = preference_matrix.shape
        self.risk_of_strategic_voting = 0
        self.happinesses = np.zeros(self.num_voters)
        self.svr_scheme = 'count_strategic_votes'
        self.original_preference_matrix = original_preference_matrix

    def run_non_strategic_election(self):
        """
        Run the election based on self.preference_matrix,
        then compute self.happinesses for each voter using the original_preference_matrix.
        Returns a 2-row array [ [ranking of alts], [their scores or votes] ].
        """
        scores = np.zeros(self.num_alternatives)

        if self.voting_scheme == 'plurality':
            for voter in range(self.num_voters):
                top_choice = self.preference_matrix[0, voter]
                if np.isnan(top_choice):
                    continue
                scores[int(top_choice)] += 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)
            
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.exponential_decay_happiness(
                    self.original_preference_matrix[:, voter],
                    election_ranking
                )

        elif self.voting_scheme == 'borda':
            for voter in range(self.num_voters):
                for rank, choice in enumerate(self.preference_matrix[:, voter]):
                    if 0 <= choice < self.num_alternatives:
                        scores[int(choice)] += (self.num_alternatives - rank - 1)
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)

            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.exp_decay_borda_style_happiness(
                    self.original_preference_matrix[:, voter],
                    election_ranking,
                    polarization={'win_fr': 0.2, 'lose_fr': 0.2, 'wl_importance': 2}
                )

        elif self.voting_scheme == 'anti_plurality':
            scores = np.full(self.num_alternatives, self.num_voters)
            for voter in range(self.num_voters):
                last_choice = self.preference_matrix[-1, voter]
                if not np.isnan(last_choice):
                    scores[int(last_choice)] -= 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)

            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.binary_happiness(
                    self.original_preference_matrix[:, voter],
                    election_ranking,
                    anti_plurality=True
                )

        elif self.voting_scheme == 'voting_for_two':
            scores = np.zeros(self.num_alternatives)
            for voter in range(self.num_voters):
                top_two = self.preference_matrix[:2, voter]
                for choice in top_two:
                    if not np.isnan(choice):
                        scores[int(choice)] += 1
            election_ranking = np.argsort(-scores, kind='stable')
            votes = np.sort(-scores, kind='stable').astype(int) * (-1)

       
            for voter in range(self.num_voters):
                self.happinesses[voter] = hpns.k_binary_happiness(
                    2,
                    self.original_preference_matrix[:, voter],
                    election_ranking
                )

        
        election_result = np.vstack((election_ranking, votes))
        return election_result


###############################
# Imperfect Info BTVA
###############################
class BTVA_ImperfectInfo(BTVA):
    """
    Single-round Imperfect (or Incomplete) Information ATVA.

    1) Randomly replaces some fraction of entries in the original preference matrix with NaN.
    2) Runs a normal, non-strategic election (on the partial matrix).
    3) If we call run_strategic_voting, enumerates all possible ways to fill the NaNs,
       checks strategic moves (Bullet, Compromise, Bury) for the strategic voter,
       and identifies which yields the greatest average improvement in happiness.
    """

    def __init__(self, voting_scheme, original_pref_matrix, noise_level=0.2, strategic_voter_idx=1):
        """
        Parameters
        ----------
        voting_scheme : str
            'plurality' or 'borda', etc.
        original_pref_matrix : np.ndarray
            The full, "true" preference matrix of shape (num_alternatives, num_voters).
            This is used for measuring happiness and also as our source for partial matrix generation.
        noise_level : float (0 to 1)
            Fraction of known entries to replace with NaN randomly, simulating incomplete knowledge.
        strategic_voter_idx : int
            The index of the voter who may vote strategically.
        """
        # Store the original as given
        self.original_full_matrix = original_pref_matrix.copy()
        self.num_alternatives, self.num_voters = self.original_full_matrix.shape
        self.strategic_voter_idx = strategic_voter_idx

        # Create a partial preference matrix by introducing NaNs
        partial_pref_matrix = self._randomly_introduce_nans(
            self.original_full_matrix, 
            noise_level=noise_level
        )

        super().__init__(
            voting_scheme=voting_scheme,
            preference_matrix=partial_pref_matrix,
            original_preference_matrix=self.original_full_matrix
        )

        
        self.partial_preference_matrix = self.preference_matrix.copy()

    def _randomly_introduce_nans(self, full_matrix, noise_level=0.2):
        """
        Randomly replace approximately noise_level fraction of entries with NaN.
        We do not replace the strategic voter's column with NaNs (assuming the
        strategic voter knows their own preferences).
        """
        partial = full_matrix.astype(float).copy()
        num_alts, num_voters = full_matrix.shape
        total_entries = num_alts * num_voters

        # We exclude the strategic voter's column from the random selection
        mask = np.ones((num_alts, num_voters), dtype=bool)
        mask[:, self.strategic_voter_idx] = False  # don't introduce NaNs in strategic voter's column
        candidate_positions = np.argwhere(mask)

        # Number of entries to set to NaN
        n_nan = int(noise_level * candidate_positions.shape[0])

        # Shuffle candidate_positions and pick the first n_nan
        np.random.shuffle(candidate_positions)
        positions_to_nan = candidate_positions[:n_nan]

        for (r, c) in positions_to_nan:
            partial[r, c] = np.nan

        return partial

    def run_strategic_voting(self, basic_result):
        """
        Perform the single-round strategic analysis under incomplete knowledge:
          1. Fill single-missing candidates in partial_preference_matrix.
          2. Enumerate all possible completions for columns that still have multiple missing candidates.
          3. For each completion, measure the strategic voter's happiness if they vote sincerely, bullet, compromise, or bury.
          4. Print average results and pick the best approach.
        Returns an array 'imperfect_incentives' of length num_voters that has 1 if a voter
        ended up using a strategic ballot, else 0.
        """
        print("\n=== Imperfect Information: Single-Round Strategic Analysis ===")
        print("Initial partial preference matrix (with NaNs):")
        print(self.partial_preference_matrix)

    
        partial_filled = self._fill_single_missing_candidates(self.partial_preference_matrix)
        print("\nAfter filling rows with exactly one missing candidate:")
        print(partial_filled)

        possible_completions = self._generate_all_completions(partial_filled)
        print(f"\nTotal possible completions: {len(possible_completions)}")

        # We'll accumulate total happiness for each strategy
        sum_sincere = 0.0
        sum_bullet = 0.0
        sum_compromise = 0.0
        sum_bury = 0.0

        # The strategic voter knows their own (true) preference from original_full_matrix
        # column = strategic_voter_idx
        strategic_sincere_ballot = self.original_full_matrix[:, self.strategic_voter_idx]

        sincere_values = []
        bullet_values = []
        compromise_values = []
        bury_values = []

        for i, completed_matrix in enumerate(possible_completions):
            print(len(possible_completions))
            print(f"\n--- Completion Scenario #{i+1} ---\n", completed_matrix)

            # Insert the strategic voter's sincere ballot
            scenario_sincere = completed_matrix.copy()
            scenario_sincere[:, self.strategic_voter_idx] = strategic_sincere_ballot

            # Evaluate the sincere scenario
            sincere_btva = BTVA(self.voting_scheme, scenario_sincere, self.original_full_matrix)
            result_sincere = sincere_btva.run_non_strategic_election()
            winner_sincere = result_sincere[0, 0]
            hvoter_sincere = sincere_btva.happinesses[self.strategic_voter_idx]
            sincere_values.append(hvoter_sincere)
            sum_sincere += hvoter_sincere
            print(f"Sincere scenario winner: {winner_sincere}, Strategic voter happiness: {hvoter_sincere}")

            # Evaluate bullet, compromise, bury
            bullet_h = self._apply_bullet_voting_and_evaluate(completed_matrix, strategic_sincere_ballot)
            bullet_values.append(bullet_h)

            compromise_h = self._apply_compromise_voting_and_evaluate(completed_matrix, strategic_sincere_ballot)
            compromise_values.append(compromise_h)

            bury_h = self._apply_bury_voting_and_evaluate(completed_matrix, strategic_sincere_ballot)
            bury_values.append(bury_h)

            sum_bullet += bullet_h
            sum_compromise += compromise_h
            sum_bury += bury_h

        n_scenarios = len(possible_completions)
        if n_scenarios == 0:
            print("No valid completions found (check data).")
            # Return no strategic moves
            return np.zeros(self.num_voters)
        
        sincere_values = np.array(sincere_values)
        bullet_values = np.array(bullet_values)
        compromise_values = np.array(compromise_values)
        bury_values = np.array(bury_values)

        avg_sincere = np.mean(sincere_values)
        avg_bullet = np.mean(bullet_values)
        avg_compromise = np.mean(compromise_values)
        avg_bury = np.mean(bury_values)

        imp_bullet = avg_bullet - avg_sincere
        imp_compromise = avg_compromise - avg_sincere
        imp_bury = avg_bury - avg_sincere

        print("\n=== Average Strategic Outcomes (Single Round) ===")
        print(f"Sincere:         {avg_sincere:.3f}")
        print(f"Bullet:          {avg_bullet:.3f} (improvement {imp_bullet:.3f})")
        print(f"Compromise:      {avg_compromise:.3f} (improvement {imp_compromise:.3f})")
        print(f"Bury:            {avg_bury:.3f} (improvement {imp_bury:.3f})")

        # Pick the largest positive improvement
        strategies = {
            'sincere': 0.0,
            'bullet':  imp_bullet,
            'compromise': imp_compromise,
            'bury': imp_bury
        }
        best_strategy = max(strategies, key=strategies.get)
        best_improve = strategies[best_strategy]

        if best_strategy == 'sincere' or best_improve <= 0:
            print("\nNo strategic ballot improves average happiness. Staying sincere.\n")
            used_strategy = False

            probability_of_regret = 0
            expected_regret = 0

        else:
            print(f"\nBest strategy = {best_strategy.upper()}, improvement = {best_improve:.3f}\n")
            used_strategy = True
            if best_strategy == 'bullet':
                chosen_values = bullet_values
            elif best_strategy == 'compromise':
                chosen_values = compromise_values
            elif best_strategy == 'bury':
                chosen_values = bury_values
            else:
                chosen_values = sincere_values  # fallback

            self.avg_chosen_happiness = np.mean(chosen_values)

            # Probability of regret: fraction of scenarios where chosen < sincere
            regrets = (chosen_values < sincere_values)
            probability_of_regret = np.mean(regrets)
            improvements = chosen_values - sincere_values
            negative_improvements = improvements[improvements < 0]
            if len(negative_improvements) == 0:
                expected_regret = 0.0
            else:
                
                expected_regret = np.mean(np.abs(negative_improvements))

        print(f"Probability of Regret: {probability_of_regret:.3f}")
        print(f"Expected Regret:       {expected_regret:.3f}")

        self.avg_sincere_happiness = avg_sincere
        imperfect_incentives = np.zeros(self.num_voters)
        if used_strategy:
            imperfect_incentives[self.strategic_voter_idx] = 1

        
        self.risk_of_strategic_voting = probability_of_regret

        return imperfect_incentives


    def _fill_single_missing_candidates(self, matrix_with_nans):
        """
        For each column, if exactly one candidate is missing,
        we can directly place that missing candidate.
        """
        matrix_copy = matrix_with_nans.copy()
        for voter in range(self.num_voters):
            col = matrix_copy[:, voter]
            nan_positions = np.where(np.isnan(col))[0]
            if len(nan_positions) == 1:
                # Exactly one missing candidate
                present = set(col[~np.isnan(col)].astype(int))
                fullset = set(range(self.num_alternatives))
                missing = list(fullset - present)
                if len(missing) == 1:
                    matrix_copy[nan_positions[0], voter] = missing[0]
        return matrix_copy

   
    def _generate_all_completions(self, partial_matrix):
        """
        Identify columns that have multiple NaNs and
        try all permutations of the missing candidates.
        Return a list of fully-completed preference matrices.
        """
        num_alts = self.num_alternatives
        completions_per_col = []

        for v in range(self.num_voters):
            col = partial_matrix[:, v]
            nan_positions = np.where(np.isnan(col))[0]
            if len(nan_positions) == 0:
                # Already fully known
                completions_per_col.append([col])
            else:
                present = set(col[~np.isnan(col)].astype(int))
                all_candidates = set(range(num_alts))
                missing_candidates = list(all_candidates - present)

                possible_columns = []
                # Permute the missing candidates among the NaN slots
                for perm in itertools.permutations(missing_candidates):
                    candidate_col = col.copy()
                    for idx, cand in zip(nan_positions, perm):
                        candidate_col[idx] = cand
                    possible_columns.append(candidate_col)

                completions_per_col.append(possible_columns)

        # cartesian product over all columns
        all_completions = []
        for columns_tuple in itertools.product(*completions_per_col):
            mat = np.column_stack(columns_tuple)
            all_completions.append(mat)
        return all_completions

    
    def _apply_bullet_voting_and_evaluate(self, completed_matrix, sincere_ballot):
        """
        In Borda: bullet = put exactly one candidate at index 0, all others -1.
        In Plurality: similarly put one candidate at top, rest -1 or unranked.
        Returns the maximum happiness the strategic voter can achieve
        by bullet voting on this scenario.
        """
        best_h = -999999
        num_alts = self.num_alternatives

        for c in range(num_alts):
            # Construct bullet ballot
            if self.voting_scheme == 'borda':
                bullet_pref = np.full_like(sincere_ballot, -1)
                bullet_pref[0] = c
            elif self.voting_scheme == 'plurality':
                # For 'plurality', same idea
                bullet_pref = np.full_like(sincere_ballot, -1)
                bullet_pref[0] = c
            elif self.voting_scheme == 'anti_plurality':
                bullet_pref = np.full_like(sincere_ballot, -1)
                bullet_pref[-1] = c
            elif self.voting_scheme == 'voting_for_two':
                bullet_pref = np.full_like(sincere_ballot, -1)
                bullet_pref[0] = c  # only one top choice

            test_matrix = completed_matrix.copy()
            test_matrix[:, self.strategic_voter_idx] = bullet_pref

            test_btva = BTVA(self.voting_scheme, test_matrix, self.original_full_matrix)
            test_btva.run_non_strategic_election()
            my_happiness = test_btva.happinesses[self.strategic_voter_idx]
            if my_happiness > best_h:
                best_h = my_happiness

        return best_h

    def _apply_compromise_voting_and_evaluate(self, completed_matrix, sincere_ballot):
        """
        In Borda: artificially raise some candidate to the top.
        In Plurality: simply place that 'compromise candidate' at index 0.
        Returns max happiness found among all compromise moves.
        """
        best_h = -999999
        sincere_top = sincere_ballot[0]

        all_candidates = set(range(self.num_alternatives)) - {sincere_top}
        for c in all_candidates:
            if self.voting_scheme == 'borda':
                # find c's original index
                loc = np.where(sincere_ballot == c)[0]
                if len(loc) == 0:
                    continue
                loc = loc[0]
                comp_pref = np.delete(sincere_ballot, loc)
                comp_pref = np.insert(comp_pref, 0, c)
            elif self.voting_scheme == 'plurality':
                # For 'plurality', just put c at 0
                comp_pref = sincere_ballot.copy()
                comp_pref = comp_pref[comp_pref != c]
                comp_pref = np.insert(comp_pref, 0, c)
            elif self.voting_scheme == 'anti_plurality':
                idx_c = np.where(sincere_ballot == c)[0]
                if len(idx_c) == 0:
                    continue
                idx_c = idx_c[0]
                comp_pref = np.delete(sincere_ballot, idx_c)
                comp_pref = np.insert(comp_pref, 0, c)

            elif self.voting_scheme == 'voting_for_two':
                idx_c = np.where(sincere_ballot == c)[0]
                if len(idx_c) == 0:
                    continue
                idx_c = idx_c[0]
                comp_pref = np.delete(sincere_ballot, idx_c)
                comp_pref = np.insert(comp_pref, 0, c)
               

            test_matrix = completed_matrix.copy()
            test_matrix[:, self.strategic_voter_idx] = comp_pref
            test_btva = BTVA(self.voting_scheme, test_matrix, self.original_full_matrix)
            test_btva.run_non_strategic_election()
            my_happiness = test_btva.happinesses[self.strategic_voter_idx]
            if my_happiness > best_h:
                best_h = my_happiness

        return best_h

    def _apply_bury_voting_and_evaluate(self, completed_matrix, sincere_ballot):
        """
        In Borda: artificially push some rival candidate c to the bottom.
        In Plurality: do a similar 'move c to last' approach.
        Returns max happiness found.
        """
        best_h = -999999
        sincere_bottom = sincere_ballot[-1]

        all_candidates = set(range(self.num_alternatives)) - {sincere_bottom}
        for c in all_candidates:
            if self.voting_scheme == 'borda':
                loc = np.where(sincere_ballot == c)[0]
                if len(loc) == 0:
                    continue
                loc = loc[0]
                bury_pref = np.delete(sincere_ballot, loc)
                bury_pref = np.append(bury_pref, c)
            elif self.voting_scheme == 'plurality':
                bury_pref = sincere_ballot.copy()
                bury_pref = np.delete(bury_pref, np.where(bury_pref == c))
                bury_pref = np.append(bury_pref, c)

            elif self.voting_scheme == 'anti_plurality':
                idx_c = np.where(sincere_ballot == c)[0]
                if len(idx_c) == 0:
                    continue
                idx_c = idx_c[0]
                bury_pref = np.delete(sincere_ballot, idx_c)
                bury_pref = np.append(bury_pref, c)
            
            elif self.voting_scheme == 'voting_for_two':
                idx_c = np.where(sincere_ballot == c)[0]
                if len(idx_c) == 0:
                    continue
                idx_c = idx_c[0]
                bury_pref = np.delete(sincere_ballot, idx_c)
                bury_pref = np.append(bury_pref, c)

            test_matrix = completed_matrix.copy()
            test_matrix[:, self.strategic_voter_idx] = bury_pref
            test_btva = BTVA(self.voting_scheme, test_matrix, self.original_full_matrix)
            test_btva.run_non_strategic_election()
            my_happiness = test_btva.happinesses[self.strategic_voter_idx]
            if my_happiness > best_h:
                best_h = my_happiness

        return best_h





pref_matrix = generate_random_preferences_matrix(5,10)

print("\n=== Imperfect Information Strategic Voting ===")
btva_imperfect = BTVA_ImperfectInfo('borda', pref_matrix, noise_level=0.2)

basic_result = btva_imperfect.run_non_strategic_election()
print("Non-strategic election result:", basic_result)
print("Voter happinesses (non-strategic):", btva_imperfect.happinesses)


imperfect_incentives = btva_imperfect.run_strategic_voting(basic_result)
print("Imperfect Information Incentives:", imperfect_incentives)
