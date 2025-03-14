import numpy as np

# Calculate the strategic voting risk depending on the strategic voting risk scheme  
def get_strategic_voting_risk(btva,  strategic_voting_incentives):
    risk_of_strategic_voting =0 

    if btva.svr_scheme == 'count_strategic_votes':
        risk_of_strategic_voting = np.sum(strategic_voting_incentives) / btva.num_voters
    elif btva.svr_scheme == 'count_strategic_votes_perm':
        risk_of_strategic_voting = np.sum(strategic_voting_incentives) / btva.get_ordering_permutation()

    return risk_of_strategic_voting

