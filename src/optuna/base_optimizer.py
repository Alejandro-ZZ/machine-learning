import abc
import logging
import time
from dataclasses import dataclass

import numpy
import optuna
import pandas

from numpy.typing import NDArray
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from sklearn.base import BaseEstimator
from sklearn.base import clone as sklearn_clone
from sklearn.base import is_classifier as sklearn_is_classifier
from sklearn.metrics import get_scorer as sklearn_get_scorer
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union
)


logger = logging.getLogger("OptunaOptimizer")



# TODO: Model parameter dependencies logic and transformations as a module-level class thought for non-code users in a interactive optimization app.
class OptunaParamSpec:
    """
    Represents a hyperparameter specification for Optuna optimization.

    Notes
    -----
    *   Using numerical distribution is better to take the best advantage of the sampler algorithms. 
        Only use categorical distributions for non-related parameter values (e.g., string options). 
        If you have ordered numerical data, but it's sparse and non-uniform, the best practice is to have 
        the sampler choose from the indices of the list (which is a simple integer range) and then you 
        map that index back to the actual value. Example::

            real_options = [0, 1, 2, 4, 8, 12, 24, 48, 72, 96, 120, 144, 168]
            option_index = trial.suggest_int("param_index", 0, len(real_options) - 1)
            value = real_options[option_index]
    
    *   If the parameter space have mixed types (e.g., strings and integers) the best approach is to create 
        a two-step conditional choice::

            param_type = trial.suggest_categorical("param_type", ["int", "str"])
            if param_type == "int":
                value = trial.suggest_int("param_value", 0, 100)
            else:
                value = trial.suggest_categorical("param_value", ["option1", "option2", "option3"])


    Parameters
    ----------
    name : str
        Hyperparameter name.
    
    type : str
        Hyperparameter type. Supported types are: "int", "float", "categorical".
    
    options : Dict[str, Any]
        Mapping with hyperparameter sampling options. Required keys depend on the hyperparameter type:

        * For "int" and "float" types: "low", "high", "step", "log".
        * For "categorical" type: "choices".
    
    default : Any, optional
        Default value for the hyperparameter.
    
    depends_on : Sequence[str], optional
            Names of parameters that must be resolved before this parameter. Dependencies determine
            the suggestion order and make a parameter inactive when any dependency is inactive.

    transform_fnc : Callable[[optuna.Trial, Any], Any], optional
        Transformation function that takes an Optuna Trial object and the suggested hyperparameter value, 
        and returns a transformed value for the hyperparameter. This can be used to apply custom 
        transformations to the suggested values before they are used in the objective function.

        Note that you can use the ``trial.params`` attribute which is a dictionary mapping parameter names 
        to their current suggested values. 
    """
    def __init__(
            self, 
            name: str, 
            type: str,
            options: Dict[str, Any],
            *,
            default: Optional[Any] = None,
            depends_on: Optional[Sequence[str]] = None,
            transform_fnc: Optional[Callable[[optuna.Trial, Any], Any]] = None,
    ):
        self.name = name
        self.type = type
        self.options = options
        self.default = default
        self.depends_on = tuple(depends_on) if depends_on is not None else tuple()
        self.transform_fnc = transform_fnc

        if (transform_fnc is not None) and (not callable(transform_fnc)):
            raise ValueError("The `transform_fnc` argument must be callable.")


    def suggest(self, trial: optuna.Trial) -> Any:
        """Suggests a hyperparameter value based on the current Optuna Trial object."""
        # Suggest the hyperparameter value based on its type and options
        if self.type == "int":
            suggested_value = trial.suggest_int(name=self.name, **self.options)
        elif self.type == "float":
            suggested_value = trial.suggest_float(name=self.name, **self.options)
        elif self.type == "categorical":
            suggested_value = trial.suggest_categorical(name=self.name, **self.options)
        else:
            raise ValueError(f"Unsupported hyperparameter type: {self.type}")

        # Apply transformation function if provided
        if self.transform_fnc is not None:
            old_value = suggested_value
            suggested_value = self.transform_fnc(trial, suggested_value)
            logger.debug(f"Parameter '{self.name}' transformed: {old_value} --> {suggested_value}")
        
        return suggested_value


class BaseOptunaOptimizer(abc.ABC):
    """
    Note
    ----
    *   multi-objective optimization: There are some restricctions when using multi-objective optimization like pruning.
    

    Parameters
    ----------
    params : List[OptunaParamSpec]
        Hyperparameter search space to optimize.

    metrics : Dict[str, str]
        Mapping of objective metric names to optimize and the corresponding optimization 
        direction: "minimize" or "maximize". For instance, given the metrics:: 
        
            metrics = {"f1-score": "maximize", "overfit_index": "minimize"}
        
        This indicates that the optimizer should maximize the ``f1-score`` metric score while 
        minimizing the ``overfit_index`` metric score (i.e., a multi-objective optimization problem).
    """
    def __init__(self, params: List[OptunaParamSpec], metrics: Dict[str, str]):
        self.parameters = self._validate_params(params) # Parameter specifications to optimize
        self.metric_names: List[str] = []               # Metric names to optimize
        self.metric_directions: List[str] = []          # Metric optimization directions (minimize or maximize)
        self.study: Optional[optuna.Study] = None       # Optuna Study instace for optimization process
        self.metadata: Dict[str, Any] = {
            "study": None,      # name, storage, metric names, directions, pruner name, sampler name
            "optimize": None,   # execution time, trials info, best trial(s) info
        }
        
        # Populate metric names and directions ordered 
        metrics = self._validate_metrics(metrics)
        for metric_name, metric_direction in metrics.items():
            self.metric_names.append(metric_name)
            self.metric_directions.append(metric_direction)

    @staticmethod
    def _validate_params(params: List[OptunaParamSpec]) -> Tuple[OptunaParamSpec, ...]:
        logger.debug("[Start] _validate_params()")
        
        # Check data type and non-empty list
        if not isinstance(params, list):
            raise ValueError("`params` must be a list of `OptunaParamSpec`.")
        if not params:
            raise ValueError("`params` must contain at least one `OptunaParamSpec`.")
        if not all(isinstance(param, OptunaParamSpec) for param in params):
            param_types = [type(param) for param in params]
            raise ValueError(f"Elements in `params` must be `OptunaParamSpec` instances. Got: {param_types}.")

        # Check for unique parameter names
        parameter_names: set[str] = set()
        for parameter in params:
            if not isinstance(parameter.name, str) or not parameter.name:
                raise ValueError("Every parameter name must be a non-empty string.")
            if parameter.name in parameter_names:
                raise ValueError(f"Duplicate parameter name: '{parameter.name}'.")
            parameter_names.add(parameter.name)

        # Check that dependencies are valid
        for parameter in params:
            if not all(isinstance(name, str) and name for name in parameter.depends_on):
                raise ValueError(f"Dependencies for parameter '{parameter.name}' must be non-empty strings.")

            dependencies = set(parameter.depends_on)
            if len(dependencies) != len(parameter.depends_on):
                raise ValueError(f"Parameter '{parameter.name}' contains duplicate dependencies.")

            missing_dependencies = dependencies - parameter_names
            if missing_dependencies:
                raise ValueError(
                    f"Parameter '{parameter.name}' depends on unknown parameter(s): "
                    f"{sorted(missing_dependencies)}."
                )

        logger.debug("[Finish] _validate_params()")
        return tuple(params)

    @staticmethod
    def _validate_metrics(metrics: Dict[str, str]) -> Dict[str, str]:
        """Validates the input arguments for the optimizer."""
        logger.debug("[Start] _validate_metrics()")

        # Optimization metrics
        # ---------------------------------
        # Main container type
        if not isinstance(metrics, dict):
            raise ValueError("`metrics` must be a dictionary mapping objective metric names to optimization directions.")

        if not metrics:
            raise ValueError("`metrics` must contain at least one objective metric.")

        # Container element values
        for metric_name, direction in metrics.items():
            if not isinstance(metric_name, str) or not metric_name:
                raise ValueError("Metric names must be non-empty strings.")
            if direction not in {"minimize", "maximize"}:
                raise ValueError(f"Direction for metric '{metric_name}' must be either 'minimize' or 'maximize'.")
    
        logger.debug("[Finish] _validate_metrics()")
        return metrics


    @abc.abstractmethod
    def supports_pruning(self) -> bool:
        """Returns whether pruning is supported for the current optimizer."""
        raise NotImplementedError()

    @abc.abstractmethod
    def create_model(self, suggestions: Dict[str, Any]) -> Any:
        """Creates and returns a model instance based on the suggested parameter values."""
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate_model(self, model: Any, suggestions: Dict[str, Any]) -> Dict[str, float]:
        """Train and evaluates a model using the resolved suggestions for the current trial."""
        raise NotImplementedError()


    def setup_study(
            self,
            name: Optional[str] = None,
            storage_url: Optional[str] = None,
            sampler: Optional[BaseSampler] = None,
            pruner: Optional[BasePruner] = None,
            load_if_exists: bool =False
    ):
        """
        Sets up an Optuna study for the optimization process.

        Parameters
        ----------
        name : str, optional
            The name of the study. If None (default), a unique name is generated automatically.
        
        storage_url : str, optional
            An URL for RDB backend storage. If None (default), an in-memory storage is used. 
            See: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html#optuna.storages.RDBStorage
            Examples:
            
                - SQLite: "sqlite:///example.db" (not recommended for parallel optimization)
                - PostgreSQL: "postgresql://user:password@localhost/dbname"
                - MySQL: "mysql://root@localhost/dbname"
        
        sampler : optuna.samplers.BaseSampler, optional
            An Optuna sampler instance to use for sampling hyperparameter values.
            If None (default), ``TPESampler`` is used during single-objective optimization and ``NSGAIISampler`` 
            during multi-objective optimization.

        pruner : optuna.pruners.BasePruner, optional
            An Optuna pruner instance to use for pruning unpromising trials. 
            If None (default), ``MedianPruner`` is used.
        
        load_if_exists : bool, optional
            Flag to control the behavior to handle a conflict of study names. In the case where a study named 
            already exists in the storage, a ``DuplicatedStudyError`` is raised if ``load_if_exists=False``. 
            Otherwise, the creation of the study is skipped, and the existing one is returned.
        """
        logger.debug("[Start] setup_study()")
        
        # Warn and disable pruning if multi-objective optimization is detected
        multi_objective = len(self.metric_names) > 1
        if pruner is not None:
            if not self.supports_pruning():
                logger.warning("Pruning is not supported by the current optimizer. Ignoring the provided pruner.")
                pruner = None
            elif multi_objective:
                logger.warning("Pruning is not supported for multi-objective optimization. Ignoring the provided pruner.")
                pruner = None

        # Get the optimization direction(s) and metric name(s)
        if not multi_objective:
            direction = self.metric_directions[0]
            directions = None
            metric_names = list(self.metric_names)
        else:
            direction = None
            directions = self.metric_directions
            metric_names = self.metric_names

        # Create or load the study
        self.study = optuna.create_study(
            study_name=name,
            storage=storage_url,
            sampler=sampler,
            pruner=pruner,
            direction=direction,
            directions=directions,
            load_if_exists=load_if_exists
        )

        # Names each dimension of the returned values of the objective function
        # Experimental in optuna
        self.study.set_metric_names(metric_names=metric_names)
        
        # Include study info in the optimizer metadata
        self.metadata["study"] = {
            "name": name,
            "storage": storage_url,
            "metrics": metric_names,
            "directions": self.metric_directions,
            "pruner": pruner.__class__.__name__ if pruner is not None else None,
            "sampler": sampler.__class__.__name__ if sampler is not None else None,
        }
        logger.debug("[Finish] setup_study()")

    def on_trial_start(self, trial: optuna.Trial):
        """
        Hook called at the start of each trial. 

        Parameters
        ----------
        trial : optuna.Trial
            The current Optuna trial object. Suggested parameter values can 
            be accessed via ``trial.params`` attribute.
        """
        pass

    def on_trial_end(self, trial: optuna.Trial, value: Union[float, tuple]):
        """
        Hook called at the end of each trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The current Optuna trial object. Suggested parameter values can 
            be accessed via ``trial.params`` attribute.

        value : float | tuple
            The objective value(s) obtained in the current trial. Match the 
            order of the ``metric_names`` attribute of the optimizer.
        """
        pass

    def objective(self, trial: optuna.Trial) -> Union[float, Sequence[float]]:
        # TODO: Support conditional hyperparameter dependencies
        # Suggest hyperparameter values 
        trial_suggestions = {
            param.name: param.suggest(trial) 
            for param in self.parameters
        }

        # Optional hook at the start of the trial
        self.on_trial_start(trial)

        # Create and evaluate the model
        try:
            trial_model = self.create_model(trial_suggestions)
            trial_metrics = self.evaluate_model(trial_model, trial_suggestions)
        except optuna.TrialPruned as e:
            logger.info(f"Trial {trial.number} pruned: {e}")
            raise e
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise e

        # Check for the required metrics
        missing_metrics = set(self.metric_names) - set(trial_metrics.keys())
        if missing_metrics:
            raise ValueError(f"Missing required metrics in trial evaluation: {missing_metrics}. Got: {trial_metrics}")

        # Extract the objective values in the correct order
        objective_value = tuple(trial_metrics[metric_name] for metric_name in self.metric_names)
        if len(objective_value) == 1:
            # Single metric value for single-objective optimization
            objective_value = objective_value[0]

        # Attach remaining metrics to trial attributes
        for metric_name, metric_value in trial_metrics.items():
            if metric_name not in self.metric_names:
                trial.set_user_attr(metric_name, metric_value)
        
        # Optional hook at the end of the trial
        self.on_trial_end(trial, objective_value)
        
        return objective_value

    def optimize(self, n_trials: Optional[int] = None, timeout: Optional[int] = None, n_jobs: int = 1):
        """
        Executes the optimization process using the configured study and objective function.

        Notes
        -----
        *   The study continues to create trials until the number of trials reaches ``n_trials``, ``timeout```
            period elapses, ``Study.stop()`` is called, or a termination signal such as SIGTERM or Ctrl+C is received.
            ``optuna.study.MaxTrialsCallback`` can ensure how many times trials will be performed across all processes.

        *   ``n_jobs`` allows parallelization using threading and may suffer from Python's GIL. 
            It is recommended to use process-based parallelization<distributed> if objective function }
            is CPU bound.

        Parameters
        ----------
        n_trials : int, optional
            The maximum number of trials to run. If None (default), there is no limit on the number of trials.
        
        timeout : int, optional
            The maximum time in seconds to run the optimization. If None (default), there is no time limit.
        
        n_jobs : int, optional
            The number of parallel jobs to run. If 1 (default), the optimization runs sequentially. 
            If -1, all available CPU cores are used. 
        """
        logger.debug("[Start] optimize()")

        # Validate that the study has been initialized
        if self.study is None:
            raise ValueError("Study is not initialized. Use `setup_study()` to create or load a study.")
        logger.info(
            f"Starting optimization: study='{self.study.study_name}', "
            f"n_trials={n_trials}, timeout={timeout}, n_jobs={n_jobs}"
        )

        # Execute the optimization process
        t0 = time.perf_counter()
        self.study.optimize(func=self.objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
        process_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - t0))
        logger.info(f"Optimization completed. Time: {process_time}")

        # Detailed trials states: completed, pruned, failed
        trials_info: Dict[str, int] = {"completed": 0, "pruned": 0, "failed": 0}
        all_trials = self.study.trials
        for trial in all_trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials_info["completed"] += 1
            elif trial.state == optuna.trial.TrialState.PRUNED:
                trials_info["pruned"] += 1
            elif trial.state == optuna.trial.TrialState.FAIL:
                trials_info["failed"] += 1
        trials_info["total"] = len(all_trials)
        logger.info(f"Trial information: {trials_info}")

        # Get the best trial(s) based on the optimization strategy
        best_trials: List[optuna.trial.FrozenTrial] = []
        if len(self.metric_names) == 1:
            best_trials.append(self.study.best_trial)
        else:
            best_trials = self.study.best_trials

        # Prepare best trial information for metadata
        best_trial_info: List[Dict[str, Any]] = []
        for trial in best_trials:
            trial_info = {
                "number": trial.number,
                "values": trial.values if len(self.metric_names) > 1 else [trial.value],
                "params": trial.params,
                "attributes": trial.user_attrs,
            }
            best_trial_info.append(trial_info)

        # Update optimizer metadata with final study statistics
        self.metadata["optimization"] = {
            "time": process_time,
            "trials": trials_info,
            "best": best_trial_info,
        }

        logger.debug("[Finish] optimize()")


