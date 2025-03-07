import numpy as np
import math

## binary_happiness for plurality and antiplurality elections where the voter only cares about their 1st or last choice 
def binary_happiness(voter_preference, election_ranking, anti_plurality=False):
    voter_preference = np.asarray(voter_preference)
    election_ranking = np.asarray(election_ranking)
    winner = election_ranking[0]

    if anti_plurality == False:
        happiness = 1 if  winner == voter_preference[0] else 0
    else:
        happiness = 0 if winner == voter_preference[-1] else 1

    return happiness


## for election schemes like this (1,1,1,...,0,0,0) where each voter only cares if one of their first k choices wins
def k_binary_happiness(k, voter_preference, election_ranking):
    voter_preference = np.asarray(voter_preference)
    election_ranking = np.asarray(election_ranking)
    winner = election_ranking[0]
    
    happiness = 1 if (np.where(voter_preference == winner)[0][0] < k) else 0
    return happiness


## exponentially decaying happiness function, but only caring about the winner
def exponential_decay_happiness(voter_preference, election_ranking):
    voter_preference = np.asarray(voter_preference)
    election_ranking = np.asarray(election_ranking)
    winner = election_ranking[0]
    winner_place_in_voter_preference = np.where(voter_preference == winner)[0][0]
    happiness = math.exp(-1 * winner_place_in_voter_preference)
    return np.around(happiness, decimals=2)
    

## exponentially decaying happiness function considering the whole preference list (suitable for Borda for example)
def exp_decay_borda_style_happiness(voter_preference, election_ranking, polarization={'win_fr': 0.3, 'lose_fr': 0.2, 'wl_importance': 1.5}):
    # about polarizion: 
    # win_fr shows the fraction that voter wants to win, 
    # lose_fr shows the fraction the voter wants to lose,
    # wl_importance how important it is for favorites to win compared to dislikes to lose.
    
    win_fraction = polarization['win_fr']
    lose_fraction = polarization['lose_fr']
    win_lose_importance = polarization['wl_importance']

    raw_happiness = 0
    max_possible_happiness = 0
    num_alternatives = len(voter_preference)
    for rank, alternative in enumerate(voter_preference):

        if not np.isin(alternative, election_ranking):
            continue
        else:
            # top preferences - voter's favorites
            if (rank + 1) <= math.floor((win_fraction * num_alternatives)):
                loss_of_rank = min((rank - np.where(election_ranking == alternative)[0][0]), 0)
                raw_happiness += math.exp(loss_of_rank - rank) # equivalent to [exp(loss_of_rank) * exp(-rank)]
                max_possible_happiness += math.exp(-rank)

            # least prefered - voter's dislikes
            elif (rank + 1) > math.ceil(((1 - lose_fraction) * num_alternatives)):
                gain_of_rank = min((np.where(election_ranking == alternative)[0][0]) - rank, 0)
                raw_happiness += (1/win_lose_importance) * math.exp(gain_of_rank + rank - (num_alternatives - 1))
                max_possible_happiness += (1/win_lose_importance) * math.exp(rank - (num_alternatives - 1))

            else:
                continue
        happiness = raw_happiness/max_possible_happiness

    return np.around(happiness, decimals=2)

