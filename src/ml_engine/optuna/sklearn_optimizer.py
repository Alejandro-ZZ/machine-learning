import logging
import pandas
import numpy
import time

from sklearn.base import BaseEstimator
from sklearn.base import clone as sklearn_clone
from sklearn.base import is_classifier as sklearn_is_classifier
from sklearn.metrics import get_scorer_names, get_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate as sklearn_cross_validate

from numpy.typing import ArrayLike, NDArray
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Union
)

from ml_engine.datasets.base import OptimizationData
from ml_engine.optuna.base import BaseOptunaOptimizer, OptunaParamSpec


logger = logging.getLogger("SklearnOptunaOptimizer")



# Dummy class to represent a scikit-learn estimator for type hinting purposes.
class BaseFitEstimator(BaseEstimator):
    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> Any:
        raise NotImplementedError()


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
    
    dataset : OptimizationData
        A dataset object providing training and validation data.
    """
    def __init__(
            self, 
            params: List[OptunaParamSpec],
            metrics: Dict[str, str],
            estimator: BaseEstimator,
            dataset: OptimizationData,
            *,
            cv: Optional[int] = None,
            seed: Optional[int] = None,
    ):
        super().__init__(params, metrics)
        
        # TODO: If needed, a mapping between metric names and sklearn scorer names can be included.
        # We expect that the metric names provided for the optimizer match the valid sklearn scorer string names.
        # For instance, "f1_macro" is a valid scorer name, but "F1-score" is not.
        # See valid names at: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
        self.check_scorer_names(names=set(metrics.keys()))
        
        self.estimator = estimator
        self.dataset = dataset
        # self.evaluator = evaluator
        self.cv = cv
        self.seed = seed
        self._check_attributes()

        self._estimator_params: Set[str] = set(self.estimator.get_params().keys())


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

    @staticmethod
    def check_scorer_names(names: Set[str]):
        valid_names = set(get_scorer_names())
        invalid_names = names - valid_names
        if invalid_names:
            raise ValueError(f"Invalid Scikir-learn scorer names: {invalid_names}. Valid names are: {valid_names}")

    def _check_attributes(self):
        # Cross-validation
        # -----------------------------------
        if self.cv is None:
            if not self.dataset.uses_validation():
                raise ValueError("Either `cv` must be provided or the dataset must include validation data.")
        else:
            if self.dataset.uses_validation():
                logger.warning("The `cv` argument will be ignored since dataset includes validation data.")
                self.cv = None
            else:
                if not isinstance(self.cv, int) or self.cv < 2:
                    raise ValueError(f"`cv` must be an integer greater than 1. Got: {self.cv}")

        # Estimator, Dataset, and Evaluator
        # -----------------------------------
        if not isinstance(self.estimator, BaseEstimator):
            raise ValueError(f"Estimator must be a subclass of 'sklearn.base.BaseEstimator'. Got: {type(self.estimator)}")
        if not isinstance(self.dataset, OptimizationData):
            raise ValueError(f"Dataset must be a subclass of 'OptimizationData'. Got: {type(self.dataset)}")
        # if not isinstance(evaluator, SklearnEvaluator):
        #     raise ValueError(f"`evaluator` must be a SklearnEvaluator. Got: {type(evaluator)}")
    
    def _evaluate_cross_validation(self, model: BaseEstimator, suggestions: Dict[str, Any]) -> Dict[str, float]:
        """Evaluates the model using cross-validation."""
        # x_train, y_train = self.dataset.train_data(**suggestions)
        x_train, y_train = self.dataset.optimization_data(train=True, **suggestions)
        
        cv_scores: Dict[str, NDArray] = sklearn_cross_validate(
            estimator=model,
            X=x_train,
            y=y_train,
            scoring=self.objective_names,
            cv=self.cv,
            n_jobs=1,   # TODO: Set this to 1 to avoid potential issues with parallelism in Optuna `optimize()` method
            return_train_score=False,
        )

        evaluation_metrics: Dict[str, float] = {}
        for raw_score_name, score_values in cv_scores.items():
            # Change the score name preffix from "test_" to "val_"
            # Example: "test_accuracy" -> "val_accuracy"
            score_name = raw_score_name.replace("test_", "val_", 1)

            # Store individual split scores. 
            # Example: "split0_val_accuracy", "split1_val_accuracy", etc.
            if score_name.startswith("val_") or score_name.startswith("train_"):
                for index, score_value in enumerate(score_values):
                    evaluation_metrics[f"split{index}_{score_name}"] = float(score_value)

            # Store the aggregated metrics (mean and std)
            # Example: "mean_val_accuracy", "std_val_accuracy"
            evaluation_metrics[f"mean_{score_name}"] = float(numpy.nanmean(score_values))
            evaluation_metrics[f"std_{score_name}"] = float(numpy.nanstd(score_values))

        # Include the objective metrics with the correct name to report
        # Example: "mean_val_accuracy" for the "accuracy" objective metric
        for metric_name in self.objective_names:
            evaluation_metrics[metric_name] = evaluation_metrics[f"mean_val_{metric_name}"]

        return evaluation_metrics

    def _evaluate_holdout(self, model: BaseFitEstimator, suggestions: Dict[str, Any]) -> Dict[str, float]:
        """Evaluates the model using holdout validation data."""
        # Retrieve the training and validation data
        # x_train, y_train = self.dataset.train_data(**suggestions)
        # x_val, y_val = self.dataset.evaluate_data(**suggestions)
        x_train, y_train = self.dataset.optimization_data(train=True, **suggestions)
        x_val, y_val = self.dataset.optimization_data(train=False, **suggestions)

        # Fit the model on the training data
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_time = time.perf_counter() - t0

        # Main evaluation results
        evaluation_metrics: Dict[str, float] = { "fit_time": fit_time }

        # Evaluate the model on the validation data for each metric
        for score_name in self.objective_names:
            scorer_fnc = get_scorer(score_name)
            t0 = time.perf_counter()
            score_value = scorer_fnc(model, x_val, y_val)
            evaluation_metrics[f"{score_name}_time"] = time.perf_counter() - t0
            evaluation_metrics[score_name] = float(score_value)

        return evaluation_metrics


    def supports_pruning(self) -> bool:
        return hasattr(self.estimator, "partial_fit")

    def create_model(self, suggestions: Dict[str, Any]) -> Any:
        # New unfitted estimator with the same parameters
        estimator_clone = sklearn_clone(self.estimator)

        # Filter the suggested parameters to only include those that are valid for the estimator
        model_params = {
            param_name: param_value 
            for param_name, param_value in suggestions.items()
            if param_name in self._estimator_params
        }

        # Set the valid parameters to the cloned estimator
        estimator_clone.set_params(**model_params)

        return estimator_clone

    def evaluate_model(self, model: Any, suggestions: Dict[str, Any]) -> Dict[str, float]:
        """Evaluates one trial without accessing final-test data."""
        
        evaluation_metrics: Dict[str, float] = {}
        if self.cv is None:
            # Evaluate the model on the validation data
            evaluation_metrics = self._evaluate_holdout(model, suggestions)
        else:
            # Evaluate the model using cross-validation
            evaluation_metrics = self._evaluate_cross_validation(model, suggestions)

        return evaluation_metrics



# TODO: Maybe include some data validation in the `SklearnOptunaOptimizer` class
def validate_sklearn_data(
        data_name: str,
        n_features: int,
        X: Optional[Union[NDArray, pandas.DataFrame]] = None,
        y: Optional[NDArray] = None,
        is_optional: bool = True,
        n_targets: int = 1,

):
    """
    Validate the input data for training, validation, or testing.

    Parameters
    ----------
    data_name : str
        Name of the data split (e.g., "training", "validation", "testing") for error messages.

    n_features : int
        Expected number of features (X) in the input data.

    X : numpy.ndarray | pandas.DataFrame, optional
        Feature data. Must be a 2D array with shape (n_samples, n_features). Default is None.

    y : numpy.ndarray, optional
        Target data. Must be a 1D or 2D array with shape (n_samples,) or (n_samples, n_targets). Default is None.

    is_optional : bool, optional
        Whether the data is optional (e.g., validation or test set). Default is True.

    n_targets : int, optional
        Expected number of target variables (y). Default is 1.
    """
    # Both X and y must be provided together or not at all
    if (X is None) != (y is None):
        raise ValueError(f"Features (X) and target (y) must be provided together. Got X={type(X)}, y={type(y)}")

    # If both are None, check if they are required or optional
    if (X is None) and (y is None):
        if not is_optional:
            raise ValueError(f"Missing required features (X) and target (y) data for '{data_name}'")
        else:
            # Optional data not provided, nothing to validate
            return

    # X and y must be numpy arrays
    x_types = (numpy.ndarray, pandas.DataFrame)
    if not isinstance(X, x_types) or not isinstance(y, numpy.ndarray):
        raise ValueError(
            f"Features (X) must be a numpy array or pandas DataFrame and target (y) must be a numpy array "
            f"for '{data_name}' data. Got X={type(X)}, y={type(y)}"
        )

    # Check X and y dimensions and compatibility
    if X.ndim != 2:
        raise ValueError(f"`Features (X) for '{data_name}' data must be a 2D array. Got shape: {X.shape}")
    if y.ndim != 1 and y.ndim != 2:
        raise ValueError(f"Target (y) for '{data_name}' data must be a 1D or 2D array. Got shape: {y.shape}")

    # X and y must have at least one sample
    if X.size == 0 or y.size == 0:
        raise ValueError(
            f"Features (X) and target (y) for '{data_name}' data cannot be empty arrays."
            f"Got shapes: X={X.shape}, y={y.shape}"
        )

    # X and y must have the same number of samples
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Features (X) and target (y) for '{data_name}' data must have the same number of samples."
            f"Got shapes: X={X.shape}, y={y.shape}"
        )

    # Check the number of features and targets
    if X.shape[1] != n_features:
        raise ValueError(f"'{data_name}' data must have {n_features} features. Got: {X.shape[1]}")
    if (y.ndim == 1 and n_targets != 1) or (y.ndim == 2 and y.shape[1] != n_targets):
        raise ValueError(f"'{data_name}' data must have {n_targets} targets. Got: {y.shape}")
