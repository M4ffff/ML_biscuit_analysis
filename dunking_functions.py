import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


import pymc as pm
import arviz as az



### Functions

def encode_biscuit_type(dataframe, biscuits):
    """
    Add new column to dataframe, changing biscuit names to integers. 

    Args:
        dataframe (df): dataframe containing 'biscuit' column
        biscuits (list): list of biscuit names in correct order
    """
    
    # check biscuit column exists
    if 'biscuit' in dataframe.columns:
        biscuit_column = dataframe['biscuit']
    else:
        raise ValueError("\"biscuit\" column does not exist in this dataframe.")
    
    for i, biscuit in enumerate(biscuits):
        biscuit_column = np.where([biscuit_column==biscuit], i, biscuit_column)[0]
        
    # set values to ints
    dataframe['encoded biscuit'] = pd.to_numeric(biscuit_column)
    
    print("added encoded biscuit column")
    
        
def washburn(gamma,phi,eta,r,t):
    """
    Function to calculate L using the washburn equation. 

    Args:
        gamma (float): tea surface tension, in N m^(-1).
        phi (float): contact angle between the biscuit and the tea surface, in rad.
        eta (float):  tea dynamic viscosity, in Pa s
        r (float): radius of the pore, in m
        t (float): time after initial dunking that the measurement was made, in s.

    Returns:
        L (float): distance up the biscuit that the tea was visible, in m.
    """
    
    numerator = gamma*r*t*np.cos(phi)
    denominator = 2*eta
    
    L = np.sqrt(numerator / denominator)
    
    return L
def run_pymc(data, gamma=6.78e-2, phi=1.45, eta=9.93e-4, rlow=1.5e-7, rhigh=1.2e-6):
    """
    Determine value of pore radius required for Washburn model to fit the real data

    Args:
        data (df): time-resolved data that Washburn model will be fit to
        gamma (float): tea surface tension. Defaults to 6.78e-2.
        phi (float): contact angle between the biscuit and the tea surface. Defaults to 1.45.
        eta (float):  tea dynamic viscosity. Defaults to 9.93e-4.
        rlow (float): lower bound of possible radius values. Defaults to 1.5e-7.
        rhigh (float): upper bound of possible radius values. Defaults to 1.2e-6.

    Returns:
        float: mean of determined distribution of pore radius. 
    """
    
    with pm.Model() as model:
        gamma=gamma
        r = pm.Uniform('r', rlow, rhigh)
        phi = phi
        eta = eta
        
        L = pm.Normal('t', 
                    mu=washburn(gamma,phi,eta,r,data["t"]), 
                    sigma=data['dL'], 
                    observed=data['L'])
        
        trace = pm.sample(1000, tune=1000, chains=10, progressbar=True)
        

    az.plot_trace(trace, var_names=["r"])
    plt.tight_layout()
    plt.show()
    
    return float(trace.posterior["r"].mean())
def calc_prob_in_distribution(mean, std, value):
    """
    Calculate probability of a given value to occur in a normal distribution

    Args:
        mean (float): Mean of normal distribution
        std (float): Std of normal distribution
        value (float): Value to have probability calculated 
    Returns:
        percentage (float): Probability of value or a more extreme value occuring in the provided distribution
    """
    
    if value > mean:
        cum_prob = 1 - norm(mean, std).cdf(value)
    elif value < mean:
        cum_prob = norm(mean, std).cdf(value)
        
    # two-tailed test
    # and converted to a percentage
    percentage = 100*2*cum_prob
    
    return percentage
def calc_num_stds(mean, std, pymc_r):
    """
    Calculate number of standard deviations a given value is from the mean of a normal distribution

    Args:
        mean (float): Mean of normal distribution
        std (float): Std of normal distribution
        value (float): Value to have probability calculated 
    Returns:
        number_stds (float): number of standard deviations a given value is from the mean 
    """
    
    number_stds = abs( pymc_r - mean) /std
    
    return number_stds

