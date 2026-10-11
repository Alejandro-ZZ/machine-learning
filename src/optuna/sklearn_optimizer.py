from sklearn.base import BaseEstimator
from sklearn.base import clone as sklearn_clone
from sklearn.base import is_classifier as sklearn_is_classifier
from sklearn.metrics import get_scorer as sklearn_get_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate as sklearn_cross_validate

from optuna.base_optimizer import BaseOptunaOptimizer
from typing import (
    Any,
    Dict,
    List,
)

class SklearnOptunaOptimizer(BaseOptunaOptimizer):
    """
    A custom optimizer for scikit-learn estimators using Optuna.

    Parameters
    ----------
    params : List[OptunaParamSpec]
        Hyperparameter search space to optimize.
    
    metrics : Dict[str, str]
        Mapping of metric names to optimization directions ("minimize" or "maximize").
    
    estimator : BaseEstimator
        A scikit-learn estimator instance to optimize.
    
    evaluator : SklearnEvaluator
        Evaluator that owns in-memory tuning data and the scikit-learn evaluation strategy.
    """
    def __init__(
            self, 
            params: List[OptunaParamSpec],
            metrics: Dict[str, str],
            estimator: BaseEstimator,
            evaluator: SklearnEvaluator
    ):
        super().__init__(params, metrics)

        if not isinstance(estimator, BaseEstimator):
            raise ValueError(f"Estimator must be a subclass of 'sklearn.base.BaseEstimator'. Got: {type(estimator)}")
        if not isinstance(evaluator, SklearnEvaluator):
            raise ValueError(f"`evaluator` must be a SklearnEvaluator. Got: {type(evaluator)}")

        self.estimator: BaseEstimator = estimator
        self.evaluator: SklearnEvaluator = evaluator

    @property
    def cv_results_(self) -> pandas.DataFrame:
        df = pandas.DataFrame()
        
        if self.study is not None:
            # Get the study trials as a DataFrame
            df = self.study.trials_dataframe()

            if not df.empty:
                # Include the hyperparameters as a single dictionary column
                param_names = [param.name for param in self.parameters]
                df["params"] = df.apply(
                    lambda row: {model_param: row["params_" + model_param] for model_param in param_names},
                    axis=1
                )

                # Rename columns to remove "user_attrs_" prefix and replace "params_" prefix by "param_"
                rename_mapping = {
                    col_name: col_name.replace("user_attrs_", "")
                    for col_name in df.columns
                    if col_name.startswith("user_attrs_")
                }
                rename_mapping.update({
                    "params_" + param_name: "param_" + param_name
                    for param_name in param_names
                })
                df.rename(columns=rename_mapping, inplace=True)

        return df


    def supports_pruning(self) -> bool:
        return hasattr(self.estimator, "partial_fit")

    def create_model(self, suggestions: Dict[str, Any]) -> Any:
        # New unfitted estimator with the same parameters
        estimator_clone = sklearn_clone(self.estimator)

        # Filter the suggested parameters to only include those that are valid for the estimator
        valid_params = estimator_clone.get_params()
        model_params = {
            param_name: param_value 
            for param_name, param_value in suggestions.items()
            if param_name in valid_params
        }

        # Set the valid parameters to the cloned estimator
        estimator_clone.set_params(**model_params)

        return estimator_clone

    def evaluate_model(
            self,
            model: Any,
            suggestions: Mapping[str, Any]
    ) -> Dict[str, float]:
        """Evaluates one trial without accessing final-test data."""
        return self.evaluator.evaluate(model, suggestions, self.metric_names)
