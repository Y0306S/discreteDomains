from openbox.core.generic_advisor import Advisor
import numpy as np

from openbox.utils.history import History
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from ConfigSpace import Configuration
from openbox.core.base import build_acq_func, build_surrogate
from mindware.components.optimizers.acq_optimizer.build import build_my_acq_optimizer
from mindware.components.optimizers.utils import MyConfiguration

class MosaicAdvisor(Advisor):

    def __init__(self, *args, **kwargs):
        # rand_prob是多大概率直接随机采样推荐，而不是用acquisition function优化得到的结果
        super().__init__(*args, **kwargs)
        self.rand_prob = 0  # 一定是0

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
            func_str=self.acq_optimizer_type, config_space=self.config_space, rng=self.rng, rand_prob=0)

    def get_suggestion(self, history: History = None, return_list: bool = True, **kwargs):
        """
        Generate a configuration (suggestion) for this query.
        Returns
        -------
        A configuration.
        """
        candidates = super().get_suggestion(history, return_list=True)
        tar_algo = kwargs.get('tar_algo', None)
        assert tar_algo is not None, "Target algorithm is not specified."

        # 根据顺序，返回第一个algorithm是tar_algo的配置

        if history is None:
            history = self.history
        for config in candidates:
            if config['algorithm'] == tar_algo and config not in history.configurations:
                return config

        # 如果都没有，则返回第一个
        for config in candidates:
            if config['algorithm'] == tar_algo:
                return config

        print(f"No configuration found for algorithm {tar_algo}.")

        KD = KnowledgeDatabase.instance()
        cs = KD.bo_history_mab[tar_algo].config_space
        config = cs.sample_configuration(1)[0]
        config_dict = config.get_dictionary()
        config_dict['algorithm'] = tar_algo
        config = MyConfiguration(Configuration(self.config_space, values=config_dict))
        return config