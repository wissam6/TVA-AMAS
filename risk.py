import numpy as np
import math

## using average gain as an indicator for risk (susceptibility) of strategic voting
def average_gain_risk(voter_strategic_gains, only_consider_gainers=True):
    voter_strategic_gains = np.asarray(voter_strategic_gains)
    non_zero_gains = voter_strategic_gains[voter_strategic_gains != 0]
    if len(non_zero_gains) == 0:
        risk = 0
    else:
        if only_consider_gainers:
            risk = np.mean(non_zero_gains)
        else:
            risk = np.mean(voter_strategic_gains)        
    return risk

## using gain percentile as an indicator for risk/ percentile=50 is equivalent to getting the median 
def gain_percentile_risk(voter_strategic_gains, percentile=75, only_consider_gainers=True):
    voter_strategic_gains = np.asarray(voter_strategic_gains)
    non_zero_gains = voter_strategic_gains[voter_strategic_gains != 0]
    if len(non_zero_gains) == 0:
        risk = 0
    else:
        if only_consider_gainers:
            risk = np.percentile(non_zero_gains, percentile)
        else:
            risk = np.percentile(voter_strategic_gains, percentile)        
    return risk

## using maximum gain as an indicator for risk
def max_gain_risk(voter_strategic_gains):
    voter_strategic_gains = np.asarray(voter_strategic_gains)
    return np.max(voter_strategic_gains)

## using proportion of voters with incentive to vote strategically as an indicator for risk
def incentive_based_risk(voter_strategic_gains):
    voter_strategic_gains = np.asarray(voter_strategic_gains)
    return np.count_nonzero(voter_strategic_gains) / len(voter_strategic_gains)