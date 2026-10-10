import abc
import logging
import optuna
import time

from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union
)


logger = logging.getLogger("OptunaOptimizer")


class OptunaParamSpec:
    pass


class BaseOptunaOptimizer(abc.ABC):
    """
    Note
    ----
    *   multi-objective optimization: There are some restricctions when using multi-objective optimization like pruning.
    

    Parameters
    ----------
    params : List[OptunaParamSpec]
        A list of `OptunaParamSpec` defining the hyperparameter search space to optimize.

    metrics : Dict[str, str]
        Mapping of objective metric names to optimize and the corresponding optimization 
        direction: "minimize" or "maximize". For instance, given the metrics:: 
        
            metrics = {"f1-score": "maximize", "overfit_index": "minimize"}
        
        This indicates that the optimizer should maximize the ``f1-score`` metric score while 
        minimizing the ``overfit_index`` metric score (i.e., a multi-objective optimization problem).
    """
    def __init__(self, params: List[OptunaParamSpec], metrics: Dict[str, str]):
        # Validate inputs
        self._validate_inputs(params, metrics)

        self.parameters: List[OptunaParamSpec] = params # Parameters to optimize
        self.metric_names: List[str] = []               # Metric names to optimize
        self.metric_directions: List[str] = []          # Metric optimization directions (minimize or maximize)
        self.study: Optional[optuna.Study] = None       # Optuna Study instace for optimization process
        self.metadata: Dict[str, Any] = {
            "study": None,              # name, storage, metric names, directions, pruner name, sampler name
            "optimize": None,           # execution time, trials info, best trial(s) info
        }
        
        # Populate metric names and directions
        for metric_name, metric_direction in metrics.items():
            self.metric_names.append(metric_name)
            self.metric_directions.append(metric_direction)


    def _validate_inputs(self, params: List[OptunaParamSpec], metrics: Dict[str, str]):
        """Validates the input arguments for the optimizer."""
        logger.debug("[Start] _validate_inputs()")
        
        # Search space parameters
        # ---------------------------------
        # Main container type
        if not isinstance(params, list):
            raise ValueError("`params` must be a list of `OptunaParamSpec`.")
        
        # Container element types
        param_types = [type(param) for param in params]
        if not all(issubclass(param_type, OptunaParamSpec) for param_type in param_types):
            raise ValueError(f"Elements in `params` must be `OptunaParamSpec` instances. Got: {param_types}.")

        # Empty parameters
        if len(params) == 0:
            raise ValueError("`params` must contain at least one `OptunaParamSpec`.")
        
        
        # Optimization metrics
        # ---------------------------------
        # Main container type
        if not isinstance(metrics, dict):
            raise ValueError("`metrics` must be a dictionary mapping objective metric names to optimization directions.")
        
        # Container element values
        for metric_name, direction in metrics.items():
            if direction not in {"minimize", "maximize"}:
                raise ValueError(f"Direction for metric '{metric_name}' must be either 'minimize' or 'maximize'.")
    
        logger.debug("[Finish] _validate_inputs()")

    # TODO: Implement
    def _objective(self, trial: optuna.Trial) -> Union[float, Sequence[float]]:
        pass

    @abc.abstractmethod
    def supports_pruning(self) -> bool:
        """Returns whether pruning is supported for the current optimizer."""
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

    def optimize(
            self,
            n_trials: Optional[int] = None,
            timeout: Optional[int] = None,
            n_jobs: int = 1
    ):
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
        self.study.optimize(func=self._objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)
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