# Set XGBoost/OpenMP thread limits to prevent over-subscription
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"
os.environ["XGBOOST_NUM_THREADS"] = "16"

import openml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import ConfigSpace as CS

# Model Imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

class HPODynamicWrapper:
    def __init__(self, task_id: int):
        """Initializes the OpenML task and determines the problem type."""
        self.task = openml.tasks.get_task(task_id)
        self.dataset = self.task.get_dataset()
        self.X, self.y, _, _ = self.dataset.get_data(
            target=self.task.target_name, dataset_format="dataframe"
        )
        
        # OpenML tasks explicitly define if they are Supervised Classification or Regression
        self.is_classification = self.task.task_type_id == 1 
        
        # Simple holdout split for max-fidelity evaluation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def _get_model_class(self, model_name: str):
        """Dynamically routes to the correct sklearn/xgboost class."""
        routers = {
            'random_forest': (RandomForestClassifier, RandomForestRegressor),
            'xgboost': (XGBClassifier, XGBRegressor),
            'neural_network': (MLPClassifier, MLPRegressor),
            # Note: LogisticRegression is classification-only. Ridge is used as the regression equivalent.
            'logistic_regression': (LogisticRegression, Ridge) 
        }
        
        if model_name not in routers:
            raise ValueError(f"Unsupported model: {model_name}")
            
        clf_class, reg_class = routers[model_name]
        return clf_class if self.is_classification else reg_class

    def evaluate(self, model_name: str, config_space: CS.ConfigurationSpace, proposed_params: dict):
        """
        Validates the proposed parameters, trains the model, and returns the loss.
        """
        # 1. Validation Layer: Check if proposed params are legally within the ConfigSpace
        try:
            valid_config = CS.Configuration(config_space, values=proposed_params)
        except ValueError as e:
            print(f"Validation Failed for {model_name}: {e}")
            # Return a massive penalty loss if the generated config is illegal
            return 9999.0

        # 2. Model Instantiation
        ModelClass = self._get_model_class(model_name)
        model_kwargs = dict(valid_config.get_dictionary())

        # Explicitly set thread count for XGBoost to prevent it from using all cores
        if model_name == "xgboost":
            model_kwargs["n_jobs"] = 16  # Same as n_threads

        model = ModelClass(**model_kwargs)

        # 3. Full-Fidelity Training
        try:
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            
            # 4. Metric Calculation (Return as a minimization objective / loss)
            if self.is_classification:
                accuracy = accuracy_score(self.y_test, predictions)
                return 1.0 - accuracy # Error rate
            else:
                mse = mean_squared_error(self.y_test, predictions)
                return mse
                
        except Exception as e:
            print(f"Training crashed with parameters {proposed_params}: {e}")
            return 9999.0 # Penalty for a crashing configuration (e.g. OOM or invalid param combo)