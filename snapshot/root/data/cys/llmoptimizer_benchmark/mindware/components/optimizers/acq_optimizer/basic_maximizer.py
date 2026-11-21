# License: MIT
# This file is partially built on SMAC3(https://github.com/automl/SMAC3), which is licensed as follows,

# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)
# Author: Aaron Klein, Marius Lindauer

import abc
import time
import warnings
from typing import Iterable, List, Union, Tuple, Optional
import random
import scipy.optimize
import numpy as np

from openbox import logger
from openbox.acquisition_function.acquisition import AbstractAcquisitionFunction
from openbox.utils.config_space import get_one_exchange_neighbourhood, \
    Configuration, ConfigurationSpace
from openbox.utils.history import History, MultiStartHistory
from openbox.utils.util_funcs import get_types
from openbox.utils.constants import MAXINT
from mindware.components.optimizers.utils import MyConfiguration


class AcquisitionFunctionMaximizer(object, metaclass=abc.ABCMeta):
    """Abstract class for acquisition maximization.

    In order to use this class it has to be subclassed and the method
    ``_maximize`` must be implemented.

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None
    ):
        self.config_space = config_space

        if rng is None:
            logger.debug('no rng given, using default seed of 1')
            self.rng = np.random.RandomState(seed=1)
        else:
            self.rng = rng

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consisting of :class:`openbox.config_space.Configuration`.
        """
        return [t[1] for t in self._maximize(acquisition_function, history, num_points, **kwargs)]

    @abc.abstractmethod
    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        """Implements acquisition function maximization.

        In contrast to ``maximize``, this method returns an iterable of tuples,
        consisting of the acquisition function value and the configuration. This
        allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`openbox.config_space.Configuration`).
        """
        raise NotImplementedError()

    def _sort_configs_by_acq_value(
            self,
            acquisition_function,
            configs: List[Configuration]
    ) -> List[Tuple[float, Configuration]]:
        """Sort the given configurations by acquisition value

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        configs : list(Configuration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value
        """

        acq_values = acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]



class LocalSearchMaximizer(AcquisitionFunctionMaximizer):
    """Implementation of openbox's local search.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        Maximum number of iterations that the local search will perform

    n_steps_plateau_walk: int
        number of steps during a plateau walk before local search terminates

    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
    ):
        super().__init__(config_space, rng)
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk

    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        """Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs:
            Additional parameters that will be passed to the
            acquisition function

        Returns
        -------
        incumbent: np.array(1, D)
            The best found configuration
        acq_val_incumbent: np.array(1,1)
            The acquisition value of the incumbent

        """

        init_points = self._get_initial_points(
            acquisition_function, num_points, history)

        acq_configs = []
        # Start N local search from different random start points
        for start_point in init_points:
            acq_val, configuration = self._one_iter(
                acquisition_function, start_point, **kwargs)

            configuration = MyConfiguration(configuration)

            configuration.origin = "Local Search"
            if 'mcts_node_id' in start_point.extra_info:
                configuration.origin += f" (mcts_node_id: {start_point.extra_info['mcts_node_id']})"

            acq_configs.append((acq_val, configuration))

        # shuffle for random tie-break
        self.rng.shuffle(acq_configs)

        # sort according to acq value
        acq_configs.sort(reverse=True, key=lambda x: x[0])

        return acq_configs

    def _get_initial_points(self, acquisition_function, num_points, history):

        if history.empty():
            init_points = self.config_space.sample_configuration(
                size=num_points)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = history.configurations
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                acquisition_function, configs_previous_runs)
            num_configs_local_search = int(min(
                len(configs_previous_runs_sorted),
                num_points)
            )
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_configs_local_search])
            )

        return init_points

    def _one_iter(
            self,
            acquisition_function: AbstractAcquisitionFunction, 
            start_point: Configuration,
            **kwargs
    ) -> Tuple[float, Configuration]:

        incumbent = start_point
        # Compute the acquisition value of the incumbent
        acq_val_incumbent = acquisition_function([incumbent], **kwargs)[0]

        local_search_steps = 0
        neighbors_looked_at = 0
        time_n = []
        while True:

            local_search_steps += 1
            if local_search_steps % 1000 == 0:
                logger.warning(
                    "Local search took already %d iterations. Is it maybe "
                    "stuck in a infinite loop?", local_search_steps
                )

            # Get neighborhood of the current incumbent
            # by randomly drawing configurations
            changed_inc = False

            # Get one exchange neighborhood returns an iterator (in contrast of
            # the previously returned list).
            all_neighbors = get_one_exchange_neighbourhood(
                incumbent, seed=self.rng.randint(MAXINT))

            # 因为空间是以算法conditional的，比如一个random_forest配置，修改一个参数，如果改的是algorithm，会变为xgboost的默认配置，没有意义

            for neighbor in all_neighbors:
                # 去掉algorithm变化的
                if neighbor['algorithm'] != incumbent['algorithm']:
                    continue
                s_time = time.time()
                acq_val = acquisition_function([neighbor], **kwargs)
                neighbors_looked_at += 1
                time_n.append(time.time() - s_time)

                if acq_val > acq_val_incumbent:
                    # logger.debug("Switch to one of the neighbors")
                    incumbent = neighbor
                    acq_val_incumbent = acq_val
                    changed_inc = True
                    break

            if (not changed_inc) or \
                    (self.max_steps is not None and
                     local_search_steps == self.max_steps):
                if len(time_n) == 0:
                    time_n.append(0.0)
                logger.debug("Local search took %d steps and looked at %d "
                                  "configurations. Computing the acquisition "
                                  "value for one configuration took %f seconds"
                                  " on average.",
                                  local_search_steps, neighbors_looked_at,
                                  np.mean(time_n))
                break

        return acq_val_incumbent, incumbent


class RandomSearchMaximizer(AcquisitionFunctionMaximizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction

    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def _maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            _sorted: bool = False,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        """Randomly sampled configurations

        Parameters
        ----------
        acquisition_function: AbstractAcquisitionFunction
            acquisition function
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function
        **kwargs
            not used

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`openbox.config_space.Configuration`).
        """

        if num_points > 1:
            rand_configs = self.config_space.sample_configuration(
                size=num_points)
        else:
            rand_configs = [self.config_space.sample_configuration(size=1)]
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i] = MyConfiguration(rand_configs[i])
                rand_configs[i].origin = 'Random Search (sorted)'
            return self._sort_configs_by_acq_value(acquisition_function, rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i] = MyConfiguration(rand_configs[i])
                rand_configs[i].origin = 'Random Search'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]


class InterleavedLocalAndRandomSearchMaximizer(AcquisitionFunctionMaximizer):
    """Implements openbox's default acquisition function optimization.

    This acq_optimizer performs local search from the previous best points
    according, to the acquisition function, uses the acquisition function to
    sort randomly sampled configurations and interleaves unsorted, randomly
    sampled configurations in between.

    Parameters
    ----------
    config_space : ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        [LocalSearchMaximizer] Maximum number of steps that the local search will perform

    n_steps_plateau_walk: int
        [LocalSearchMaximizer] number of steps during a plateau walk before local search terminates

    n_sls_iterations: int
        [LocalSearchMaximizer] number of local search iterations

    """

    def __init__(
            self,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
            n_sls_iterations: int = 10,
            rand_prob=0.25
    ):
        super().__init__(config_space, rng)
        self.random_search = RandomSearchMaximizer(
            config_space=config_space,
            rng=rng
        )
        self.local_search = LocalSearchMaximizer(
            config_space=config_space,
            rng=rng,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk
        )
        self.n_sls_iterations = n_sls_iterations
        self.rand_prob = rand_prob

        # =======================================================================
        # self.local_search = DiffOpt(
        #     acquisition_function=acquisition_function,
        #     config_space=config_space,
        #     rng=rng
        # )
        # =======================================================================

    def maximize(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        history: openbox.utils.history.History
            history object
        num_points: int
            number of points to be sampled
        **kwargs
            passed to acquisition function

        Returns
        -------
        Iterable[Configuration]
            List of configurations.
        """

        next_configs_by_local_search = []
        # rand_prob的概率禁用local search，直接全部用random search采样
        if self.rng.rand() > self.rand_prob:
            next_configs_by_local_search = self.local_search._maximize(
                acquisition_function, history, self.n_sls_iterations, **kwargs
            )

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_search._maximize(
            acquisition_function,
            history,
            num_points - len(next_configs_by_local_search),
            _sorted=True,
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of openbox. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = (
                next_configs_by_random_search_sorted
                + next_configs_by_local_search
        )
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        logger.debug(
            "First 10 acq func (origin) values of selected configurations: %s",
            str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:10]])
        )
        challengers = [_[1] for _ in next_configs_by_acq_value]

        return challengers

    def _maximize(
            self,
            history: History,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()

