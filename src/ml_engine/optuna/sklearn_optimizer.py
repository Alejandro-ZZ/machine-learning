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
    Set
)

from ml_engine.datasets.base_dataset import OptimizationData
from ml_engine.optuna.base_optimizer import BaseOptunaOptimizer, OptunaParamSpec


logger = logging.getLogger("SklearnOptunaOptimizer")


"""
Scikit-learn classification report structure:
```
    {
        "accuracy": <float>,
        "<class_1>": { "f1-score": <float>, "precision": <float>, "recall": <float>, "support": <int> },
        "<class_2>": { ... }, 
        ...
        "macro avg": { "f1-score": <float>, "precision": <float>, "recall": <float>, "support": <int> },
        "weighted avg": { "f1-score": <float>, "precision": <float>, "recall": <float>, "support": <int> }
    }
```
"""
def sklearn_classification_report(y_true, y_pred, class_names=None, confidence_level=0.95) -> Dict[str, Any]:
    """
    Computes a classification report using scikit-learn's `classification_report` function.
    Include additional metrics in the report (per-class, macro, and weighted):

        * Jaccard index (jaccard)
    
    Include overall metrics:
        * Cohen's kappa (cohen_kappa)
        * Confidence intervals (intervals) as a list of [lower_bound, upper_bound]
    
    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        True and predicted labels.
    
    class_names : list of str, optional
        Names of the classes. If None, class names will be inferred from the unique values in `y_true`.

    confidence_level : float, optional
        Confidence level for the confidence intervals. Default is 0.95 (95% confidence interval).
    """
    from sklearn.metrics import classification_report, jaccard_score, cohen_kappa_score
    from statsmodels.stats.proportion import proportion_confint
    #
    if class_names is None:
        n_classes = len(numpy.unique(y_true))
        class_names = [str(i) for i in range(n_classes)]
    else:
        n_classes = len(class_names)
        if n_classes != len(numpy.unique(y_true)):
            raise ValueError(f"Number of class names ({n_classes}) does not match number of unique classes in y_true ({len(numpy.unique(y_true))}).")
    #
    # The support (number of true samples) for each class
    class_support = numpy.bincount(y_true)
    #
    report: dict = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=range(n_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=numpy.nan
    )
    #
    # Confidence intervals 
    ci_lower, ci_upper = proportion_confint(
        count=(y_true == y_pred).sum(),
        nobs=len(y_true),
        alpha=1 - confidence_level,
        method="wilson"
    )
    report["intervals"] = [float(ci_lower), float(ci_upper)]
    #
    # Cohen's kappa score
    cohen_kappa = cohen_kappa_score(y1=y_true, y2=y_pred)
    report["cohen_kappa"] = float(cohen_kappa)
    #
    # Jaccard score (IoU) for each class and averages.
    # Mean IoU is the macro average of the Jaccard scores.
    classes_jaccard: NDArray = jaccard_score(
        y_true=y_true,
        y_pred=y_pred,
        average=None,
        labels=range(n_classes),
        zero_division=0
    )
    # import pdb; pdb.set_trace()
    macro_jaccard = numpy.average(classes_jaccard, weights=None)
    weighted_jaccard = numpy.average(classes_jaccard, weights=class_support)
    #
    report["macro avg"]["jaccard"] = float(macro_jaccard)
    report["weighted avg"]["jaccard"] = float(weighted_jaccard)
    for class_index, class_name in enumerate(class_names):
        report[class_name]["jaccard"] = float(classes_jaccard[class_index])
    #
    return report

def display_report(report: Dict[str, Any], class_names: List[str], average: str = "macro", digits: int = 4, format: str = "grid"):
    """
    Display two formatted tables for the classification report:
    1. Per-class metrics table
    2. Overall metrics table based on the specified average type ("macro" or "weighted").
    """
    from tabulate import tabulate

    # Prepare per-class metrics table
    per_class_table = []
    for class_name in class_names:
        class_metrics = report[class_name]
        per_class_table.append({
            "Class": class_name,
            "Precision": class_metrics.get("precision", float("nan")),
            "Recall": class_metrics.get("recall", float("nan")),
            "F1-Score": class_metrics.get("f1-score", float("nan")),
            "Support": class_metrics.get("support", float("nan")),
            "Jaccard": class_metrics.get("jaccard", float("nan"))
        })

    # Other overall metric names that are not per-class or average metrics
    other_metrics = {
        name 
        for name in report.keys() 
        if ("avg" not in name) and (name not in class_names)
    }
    
    # Prepare overall metrics table 
    overall_metrics = report[f"{average} avg"]

    # Make table as | metric name | value |
    overall_table = []
    for metric_name in other_metrics.union(overall_metrics.keys()):
        metric_value = overall_metrics.get(metric_name, report.get(metric_name, float("nan")))
        #
        # Format to the specified number of digits
        if isinstance(metric_value, float):
            metric_value = round(metric_value, digits)
        elif isinstance(metric_value, (list, tuple)):
            metric_value = [round(v, digits) for v in metric_value]
        else:
            raise ValueError(f"Unexpected metric value type for '{metric_name}': {type(metric_value)}")
        #
        overall_table.append([ metric_name, metric_value ])

    # Display the tables
    print("\nPer-Class Metrics:")
    print(tabulate(
        per_class_table, 
        headers="keys", 
        tablefmt=format, 
        floatfmt=f".{digits}f"
    ))
    
    print(f"\nOverall Metrics ({average} average):")
    print(tabulate(
        overall_table, 
        headers=["Metric", "Value"], 
        tablefmt=format, 
        # floatfmt=f".{digits}f"
    ))




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
        x_train, y_train = self.dataset.train_data(**suggestions)
        
        cv_scores: Dict[str, NDArray] = sklearn_cross_validate(
            estimator=model,
            X=x_train,
            y=y_train,
            scoring=self.metric_names,
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
        for metric_name in self.metric_names:
            evaluation_metrics[metric_name] = evaluation_metrics[f"mean_val_{metric_name}"]

        return evaluation_metrics

    def _evaluate_holdout(self, model: BaseFitEstimator, suggestions: Dict[str, Any]) -> Dict[str, float]:
        """Evaluates the model using holdout validation data."""
        # Retrieve the training and validation data
        x_train, y_train = self.dataset.train_data(**suggestions)
        x_val, y_val = self.dataset.validate_data(**suggestions)

        # Fit the model on the training data
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_time = time.perf_counter() - t0

        # Main evaluation results
        evaluation_metrics: Dict[str, float] = { "fit_time": fit_time }

        # Evaluate the model on the validation data for each metric
        for score_name in self.metric_names:
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
