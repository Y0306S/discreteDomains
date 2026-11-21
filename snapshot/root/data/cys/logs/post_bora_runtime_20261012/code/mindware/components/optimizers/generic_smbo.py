# License: MIT

import time
from typing import List
from tqdm import tqdm
import numpy as np
from openbox import logger
from openbox.optimizer.base import BOBase
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT
from openbox.utils.limit import run_obj_func
from openbox.utils.util_funcs import parse_result, deprecate_kwarg
from openbox.utils.history import Observation, History
from openbox.utils.early_stop import EarlyStopException
from openbox.visualization import build_visualizer


class MySMBO(BOBase):

    @deprecate_kwarg('num_objs', 'num_objectives', 'a future version')
    @deprecate_kwarg('time_limit_per_trial', 'max_runtime_per_trial', 'a future version')
    @deprecate_kwarg('runtime_limit', 'max_runtime', 'a future version')
    def __init__(
            self,
            objective_function: callable,
            config_space,
            num_objectives=1,
            num_constraints=0,
            sample_strategy: str = 'bo',
            max_runs=100,
            max_runtime=None,
            max_runtime_per_trial=None,
            advisor_type='default',
            surrogate_type='auto',
            acq_type='auto',
            acq_optimizer_type='auto',
            initial_runs=3,
            init_strategy='random_explore_first',
            initial_configurations=None,
            ref_point=None,
            transfer_learning_history: List[History] = None,
            early_stop=False,
            early_stop_kwargs=None,
            logging_dir='logs',
            task_id='OpenBox',
            visualization='none',
            auto_open_html=False,
            random_state=None,
            logger_kwargs: dict = None,
            advisor_kwargs: dict = None,
    ):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.FAILED_PERF = [np.inf] * num_objectives
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         max_runtime=max_runtime, max_runtime_per_trial=max_runtime_per_trial,
                         sample_strategy=sample_strategy, transfer_learning_history=transfer_learning_history,
                         logger_kwargs=logger_kwargs)

        self.advisor_type = advisor_type
        advisor_kwargs = advisor_kwargs or {}
        _logger_kwargs = {'force_init': False}  # do not init logger in advisor

        if advisor_type == 'mosaic':
            from mindware.components.optimizers.advisor.mosaic_advisor import MosaicAdvisor
            self.config_advisor = MosaicAdvisor(config_space,
                                                num_objectives=num_objectives,
                                                num_constraints=num_constraints,
                                                initial_trials=initial_runs,
                                                init_strategy=init_strategy,
                                                initial_configurations=initial_configurations,
                                                optimization_strategy=sample_strategy,
                                                surrogate_type=surrogate_type,
                                                acq_type=acq_type,
                                                acq_optimizer_type=acq_optimizer_type,
                                                ref_point=ref_point,
                                                transfer_learning_history=transfer_learning_history,
                                                early_stop=early_stop,
                                                early_stop_kwargs=early_stop_kwargs,
                                                task_id=task_id,
                                                output_dir=logging_dir,
                                                random_state=random_state,
                                                logger_kwargs=_logger_kwargs,
                                                **advisor_kwargs)
        else:
            from mindware.components.optimizers.advisor.generic_advisor import MyAdvisor
            self.config_advisor = MyAdvisor(config_space,
                                            num_objectives=num_objectives,
                                            num_constraints=num_constraints,
                                            initial_trials=initial_runs,
                                            init_strategy=init_strategy,
                                            initial_configurations=initial_configurations,
                                            optimization_strategy=sample_strategy,
                                            surrogate_type=surrogate_type,
                                            acq_type=acq_type,
                                            acq_optimizer_type=acq_optimizer_type,
                                            ref_point=ref_point,
                                            transfer_learning_history=transfer_learning_history,
                                            early_stop=early_stop,
                                            early_stop_kwargs=early_stop_kwargs,
                                            task_id=task_id,
                                            output_dir=logging_dir,
                                            random_state=random_state,
                                            logger_kwargs=_logger_kwargs,
                                            **advisor_kwargs)


    def run(self) -> History:
        for idx in tqdm(range(self.iteration_id, self.max_runs)):
            if self.time_left <= 0:
                logger.info(f'max runtime ({self.max_runtime}s) exceeded, stop optimization.')
                break
            start_time = time.time()
            try:
                self.iterate(time_left=self.time_left)
            except EarlyStopException:
                logger.info(f'Early stop triggered at iter {idx}!')
                break
            runtime = time.time() - start_time
            self.time_left -= runtime
        return self.get_history()

    def iterate(self, time_left=None, **kwargs) -> Observation:
        # get configuration suggestion from advisor
        config = self.config_advisor.get_suggestion(**kwargs)

        if config in self.config_advisor.history.configurations:
            logger.warning('Evaluating duplicated configuration: %s' % config)

        if time_left is None:
            timeout = self.max_runtime_per_trial
        elif self.max_runtime_per_trial is None:
            timeout = time_left
        else:
            timeout = min(time_left, self.max_runtime_per_trial)
        if np.isinf(timeout):
            timeout = None

        # evaluate configuration on objective_function
        obj_args, obj_kwargs = (config,), dict()
        result = run_obj_func(self.objective_function, obj_args, obj_kwargs, timeout)

        # parse result
        ret, timeout_status, traceback_msg, elapsed_time = (
            result['result'], result['timeout'], result['traceback'], result['elapsed_time'])
        if timeout_status:
            trial_state = TIMEOUT
        elif traceback_msg is not None:
            trial_state = FAILED
            logger.error(f'Exception in objective function:\n{traceback_msg}\nconfig: {config}')
        else:
            trial_state = SUCCESS
        if trial_state == SUCCESS:
            objectives, constraints, extra_info = parse_result(ret)
        else:
            objectives, constraints, extra_info = self.FAILED_PERF.copy(), None, None

        # update observation to advisor
        observation = Observation(
            config=config, objectives=objectives, constraints=constraints,
            trial_state=trial_state, elapsed_time=elapsed_time, extra_info=extra_info,
        )
        self.config_advisor.update_observation(observation)

        self.iteration_id += 1
        # Logging
        if self.num_constraints > 0:
            logger.info('Iter %d, objectives: %s. constraints: %s.' % (self.iteration_id, objectives, constraints))
        else:
            logger.info('Iter %d, objectives: %s.' % (self.iteration_id, objectives))

        return observation
