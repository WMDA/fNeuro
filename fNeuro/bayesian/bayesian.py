import numpy as np
import pandas as pd


def z_score_to_bayes_factor(zscore):
    
    '''
    Function to convert z scores to 
    bayes factors

    Parameters
    ----------
    zscore: float

    Returns
    -------
    bayes factor
    '''
    return np.exp(zscore**2 / 2)

def logistic_to_bayes_factor(coeffiecent: float) -> float:
    
    '''
    Function to convert logistic regression 
    co-efficents to bayes factor

    Parameters
    ----------
    coeffiecent: float
        coeffient from logistic regression

    Returns
    -------
    bayes factor: float


    '''
    return np.exp(np.power(coeffiecent, 2) / 2)

def bf_upper_bound(p: float) -> float:
     '''
     Function to calculate the bayes factor upper bound

     1/-ep log p

     where e is natural base
     p is p value
     and log is natural log

     Parameters
     ----------
     p: float
         p value
     
     Returns
     ------
     float: bayes factor upper bound
     '''
     return 1/((-np.e * p) * np.log(p))

def fraction_to_decimal_odds(numerator: float, denominator: int) -> float:
    
    """
    Converts a fraction to decimal odds.    
    Parameters
    ---------
    numerator: float 
        Numerator of fraction
    denominator: int 
        Denominator of fraction    
    Returns
    -------
    float: The decimal odds of the fraction.
    """
    
    # Add 1 to the numerator to account for the fact that the probability of
    # winning a bet with decimal odds of x is 1/x.    
    return numerator / denominator + 1

def decimal_odds_to_percentage(decimal_odds) -> float:
    
    """
    Converts a decimal odds to a percentage.    
    Parameters
    ----------
    decimal_odds: float 
        The decimal odds to convert.    
    Returns
    -------
    percentage: float 
        The percentage equivalent of the decimal odds.
    """    
    
    percentage = 100 * (1/decimal_odds)
    return percentage

def bayes_factor_upper_bound(p):
    
    '''
    Function to calculate bayes factor upper bound
    and probability of null and alterantive hypothesis

    1/-ep log p

    where e is natural base
    p is p value
    and log is natural log

    Parameters
    -----------
    p: float
        p-value

    Returns
    -------
    dict: dictionary object
        dictionary of bayes factor upper bound
        and probabilities

    '''

    bfb = bf_upper_bound(p)
    decimal_odds = fraction_to_decimal_odds(bfb, 1) # odds are null hypothesis
    percentage = decimal_odds_to_percentage(decimal_odds)
    
    # This if else needed as probabilities for alternative
    # and null flip once p hits about 0.36. At this point
    # the bayes factor bound becomes negative for ease of interpretation
    # as the hypothesis switches 
    

    if p >= 0.36:
        alternative_prob = percentage
        null_prob = 100-percentage
        bfb = -abs(bfb)
    else:
        alternative_prob = 100-percentage
        null_prob = percentage
    return {
        'BFB': bfb,
        'null_hypothesis_probabilty': round(null_prob, 4),
        'alternative_hypothesis_probabilty': round(alternative_prob, 4),
        'odds': decimal_odds
    }


def bayesian_cluster_info(csv_file: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Function to calculate bayes factor bound, odds 
    and probability of null hypothesis

    Parameters
    ----------
    csv_file: pd.DataFrame
        DataFrame with cluster information

    Returns
    -------
    csv_file: pd.DataFrame 
        csv_file with additional bfb, odds 
        and probability of null hypothesis 
    
    '''
    
    csv_file['BFB'] = csv_file['pval'].apply(lambda p: bayes_factor_upper_bound(p)['BFB'])
    csv_file['odds'] = csv_file['pval'].apply(lambda p: bayes_factor_upper_bound(p)['odds'])
    csv_file['null_proability'] = csv_file['pval'].apply(lambda p: bayes_factor_upper_bound(p)['null_hypothesis_probabilty'])
    return csv_file
    
def decimal_to_odds(probability: float) -> float:
    
    """
    Converts a decimal probability to odds.    
    Parameters
    ----------
    probability: float 
        The decimal probability.    
    Returns:
      odds: float.
    """
    
    return probability / (1 - probability)