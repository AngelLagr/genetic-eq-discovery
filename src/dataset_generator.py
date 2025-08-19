import numpy as np
import pandas as pd

def generate_dataset(formula:str, feature_names:list, n_samples:int=100, seed:int=42):
    """
    Generate a synthetic dataset based on a mathematical formula.

    Parameters:
        formula (str): Valid Python formula, e.g. "2*x1 + np.sin(x2)"
        feature_names (list): List of variable names used in the formula
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility

    Returns:
        df (pd.DataFrame): DataFrame containing features and target y
    """
    np.random.seed(seed)

    # Randomly generate features (uniform distribution in [-5, 5])
    data = {}
    for name in feature_names:
        data[name] = np.random.uniform(-5, 5, size=n_samples)
    
    # Compute y by evaluating the formula
    local_vars = {name: data[name] for name in feature_names}
    local_vars['np'] = np  # Allow use of np functions like np.sin, np.exp, etc.

    # Evaluate the formula
    y = eval(formula, {}, local_vars)
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    df['y'] = y

    return df


def generate_time_dataset(formula:str, feature_names:list, n_samples:int=100):
    """
    Generate a time-based dataset using a mathematical formula.

    Parameters:
        formula (str): Valid Python formula, e.g. "2*x1 + np.sin(x2)"
        feature_names (list): List of variable names used in the formula (first variable is time)
        n_samples (int): Number of samples to generate

    Returns:
        df (pd.DataFrame): DataFrame containing time feature and target y
    """

    # Generate the time feature
    data = {}
    data[feature_names[0]] = np.linspace(0, 20, n_samples)

    # Compute y by evaluating the formula
    local_vars = {name: data[name] for name in feature_names}
    local_vars['np'] = np  # Allow use of np functions like np.sin, np.exp, etc.

    # Evaluate the formula
    y = eval(formula, {}, local_vars)
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    df['y'] = y

    return df