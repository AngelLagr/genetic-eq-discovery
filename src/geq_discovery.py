import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error

def load_dataset(csv_path:str, target_col='y'):
    """
    Load dataset from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file
        target_col (str): Name of the target column

    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
    """

    # Load the dataset
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

    return X, y


def train_symbolic_regressor(X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray):
    """
    Train a symbolic regressor using genetic programming.

    Parameters:
        X_train (np.ndarray): Training feature set
        y_train (np.ndarray): Training target values
        X_test (np.ndarray): Testing feature set
        y_test (np.ndarray): Testing target values

    Returns:
        est (gplearn.genetic.SymbolicRegressor): Trained symbolic regressor
    """

    # Initialize symbolic regressor
    est = SymbolicRegressor(
        population_size=1000,
        generations=80,
        tournament_size=100,
        stopping_criteria=0.01,
        const_range=(-5, 5),
        init_depth=(3, 8),
        init_method='full',
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'cos'],
        parsimony_coefficient=0.001,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        verbose=1,
        n_jobs=-1,
    )

    # Fitting
    est.fit(X_train, y_train)

    # Operate the loss
    y_pred = est.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Display the loss and the formula
    print("MSE test:", mse)
    print("formula :", est._program)

    return est