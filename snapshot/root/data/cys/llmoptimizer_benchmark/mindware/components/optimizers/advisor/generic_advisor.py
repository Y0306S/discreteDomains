from openbox.core.generic_advisor import Advisor
import numpy as np

from openbox.core.base import build_acq_func, build_surrogate
from mindware.components.optimizers.acq_optimizer.build import build_my_acq_optimizer
from mindware.components.optimizers.utils import MyConfiguration


class MyAdvisor(Advisor):

    def __init__(self, *args, **kwargs):
        # rand_prob是多大概率直接随机采样推荐，而不是用acquisition function优化得到的结果
        self.acq_optimizer_rand_prob = kwargs.get('acq_optimizer_rand_prob', 0.0)  # 多大概率在acq_optimizer中不要local search，而是直接随机采样，写在super前面是为了让setup_bo_basics能用到
        super().__init__(*args, **kwargs)
        # self.rand_prob = kwargs.get('rand_prob', 0.1)  # 多大概率直接随机采样推荐，而不是用acquisition function优化得到的结果，写在super后是因为super设置过一遍


    def setup_bo_basics(self):
        """
        Prepare the basic BO components.
        Returns
        -------
        An optimizer object.
        """
        if self.num_objectives == 1:
            self.surrogate_model = build_surrogate(func_str=self.surrogate_type,
                                                   config_space=self.config_space,
                                                   rng=self.rng,
                                                   transfer_learning_history=self.transfer_learning_history)
        elif self.acq_type == 'parego':
            func_str = 'parego_' + self.surrogate_type
            self.surrogate_model = build_surrogate(func_str=func_str,
                                                   config_space=self.config_space,
                                                   rng=self.rng,
                                                   transfer_learning_history=self.transfer_learning_history)
        else:  # multi-objectives
            self.surrogate_model = [build_surrogate(func_str=self.surrogate_type,
                                                    config_space=self.config_space,
                                                    rng=self.rng,
                                                    transfer_learning_history=self.transfer_learning_history)
                                    for _ in range(self.num_objectives)]

        if self.num_constraints > 0:
            self.constraint_models = [build_surrogate(func_str=self.constraint_surrogate_type,
                                                      config_space=self.config_space,
                                                      rng=self.rng) for _ in range(self.num_constraints)]

        if self.acq_type in ['mesmo', 'mesmoc', 'mesmoc2', 'usemo']:
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       config_space=self.config_space)
        else:
            self.acquisition_function = build_acq_func(func_str=self.acq_type,
                                                       model=self.surrogate_model,
                                                       constraint_models=self.constraint_models,
                                                       ref_point=self.ref_point)
        if self.acq_type == 'usemo':
            self.acq_optimizer_type = 'usemo_optimizer'
        self.acq_optimizer = build_my_acq_optimizer(
            func_str=self.acq_optimizer_type, config_space=self.config_space, rng=self.rng, rand_prob=self.acq_optimizer_rand_prob)


    @staticmethod
    def sample_random_configs(config_space, num_configs=1, excluded_configs=None):
        """
        Sample a batch of random configurations.

        Parameters
        ----------
        config_space: ConfigurationSpace
            Configuration space object.
        num_configs: int
            Number of configurations to sample.
        excluded_configs: optional, List[Configuration] or Set[Configuration]
            A list of excluded configurations.

        Returns
        -------
        configs: List[Configuration]
            A list of sampled configurations.
        """
        if excluded_configs is None:
            excluded_configs = set()

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = config_space.sample_configuration()
            config = MyConfiguration(config)
            config.origin = "Random Search"
            sample_cnt += 1
            if config not in configs and config not in excluded_configs:
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                print('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs


    def create_initial_design(self, init_strategy='default'):
        """
        Create several configurations as initial design.
        Parameters
        ----------
        init_strategy: str

        Returns
        -------
        Initial configurations.
        """

        valid_configs = super().create_initial_design(init_strategy=init_strategy)

        # 修改为使用MyConfiguration
        for i in range(len(valid_configs)):
            if not isinstance(valid_configs[i], MyConfiguration):
                valid_configs[i] = MyConfiguration(valid_configs[i])
                valid_configs[i].origin = "Random Search"
        return valid_configs
