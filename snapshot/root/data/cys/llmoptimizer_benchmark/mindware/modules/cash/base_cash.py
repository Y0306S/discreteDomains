import os
import json
import pickle as pkl
from typing import List, Union, Callable
from sklearn.metrics._scorer import _BaseScorer
import numpy as np
import time
import datetime

from mindware.modules.base import BaseAutoML
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode

from mindware.components.config_space.cs_builder import get_cash_cs
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import CONFIGS


class BaseCASH(BaseAutoML):
    
    name='cash'
    
    def __init__(self, include_algorithms: List[str] = None, task_type: int = None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', sub_optimizer: str = 'smac', inner_iter_num_per_iter=1,
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir='./data', seed=1, n_jobs=1, topk=50, rmfiles=False,
                 ensemble_method=None, ensemble_size=5, task_id='test',
                 filter_params=None, optimizer_params=None):

        super(BaseCASH, self).__init__(
            task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, inner_iter_num_per_iter=inner_iter_num_per_iter,
            time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=per_run_time_limit,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs, topk=topk, rmfiles=rmfiles,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size, task_id=task_id, optimizer_params=optimizer_params
        )

        if optimizer not in ['llmmab', 'llmbo', 'smac', 'llm', 'tpe', 'hebo', 'cmaes', 'pt4opt_tpe', 'pt4opt_cmaes', 'lolbo', 'pt4opt_lolbo', 'ptllm', 'ptllm_tpe', 'ptllm_smac', 'ptllm_cmaes', 'ptllm_hebo', 'random_search', 'mab', 'block_0', 'block_1', 'bopro', 'bora', 'opro', 'llambo', 'mosaic']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if sub_optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid sub_optimizer: %s for CASH!' % sub_optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.include_algorithms = include_algorithms
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        path = 'CASH-%s(%d)-%s_%s_%s' % (
            optimizer, self.seed, self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        cs_args = {
            'resampling_params': resampling_params,
            'data_node': data_node
        }
        from mindware.components.config_space.cs_builder import get_cs_args
        cs_args = get_cs_args(**cs_args)
        self.cs_args = cs_args

        self.filter_params = filter_params
        # select models
        if self.filter_params is not None and 'n_algorithm' in self.filter_params:
            n_algo = self.filter_params['n_algorithm']
            include_algorithms = self._recommand_models(self.task_type, task_id=self.task_id, data_node=self.data_node, metric=self.metric_name, n_algo=n_algo, include_algorithms=include_algorithms)

        if include_algorithms is not None and len(include_algorithms) == 1:
            from mindware.components.config_space.cs_builder import get_hpo_cs
            self.cs = get_hpo_cs(estimator_id=include_algorithms[0], task_type=self.task_type, **cs_args)
        else:
            from mindware.components.config_space.cs_builder import get_cash_cs
            self.cs = get_cash_cs(include_algorithms=include_algorithms, task_type=self.task_type, **cs_args)
            
        kd_params = {
            k: v
            for k, v in (self.optimizer_params or {}).items()
            if k not in {
                "task_type",
                "metric",
                "task_id",
                "datanode",
                "config_space",
                "include_algorithms",
                "original_data_dir",
                "rag_data_dir",
            }
        }
        KnowledgeDatabase.instance(
            task_type=task_type, metric=self.metric_name, task_id=task_id, datanode=self.data_node, config_space=self.cs, include_algorithms=include_algorithms,
            original_data_dir=CONFIGS.ORIGINAL_DATA_DIR, rag_data_dir=CONFIGS.RAG_DATA_DIR, **kd_params
        )


        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        # Define evaluator and optimizer

        data_node_path = os.path.join(self.output_dir, 'data_node.pkl')
        with open(data_node_path, 'wb') as f:
            pkl.dump(data_node, f)
        self.evaluator = None
        if self.task_type in CLS_TASKS:
            from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator
            self.evaluator = CASHCLSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node_path,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                if_imbal=self.if_imbal)
        else:
            from mindware.modules.cash.cash_evaluator import CASHRGSEvaluator
            self.evaluator = CASHRGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node_path,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed)

        self.optimizer = self.build_optimizer(name='cash', sub_optimizer=sub_optimizer)

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-CASH-task_type%d-%s(%d)' % (self.task_type, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def get_conf(self, save=False):

        conf = super(BaseCASH, self).get_conf()
        from ConfigSpace.hyperparameters import Constant
        if isinstance(self.cs['algorithm'], Constant):
            conf['include_algorithms'] = [self.cs['algorithm'].value]
        else:
            conf['include_algorithms'] = self.cs['algorithm'].choices

        if save:
            conf_to_save = dict(conf)
            # Live HF clients are not JSON-serializable; persist a path hint instead.
            opt_params = conf_to_save.get("optimizer_params")
            if isinstance(opt_params, dict) and "llm_client" in opt_params:
                opt_params = dict(opt_params)
                client = opt_params.pop("llm_client", None)
                if client is not None and "qwen_adapter_path" not in opt_params:
                    adapter = getattr(client, "adapter_path", None)
                    model_path = getattr(client, "model_path", None)
                    if adapter:
                        opt_params["qwen_adapter_path"] = adapter
                    if model_path:
                        opt_params["qwen_model_path"] = model_path
                conf_to_save["optimizer_params"] = opt_params
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf_to_save, f, indent=4, default=str)

        return conf
import os
import json
import pickle as pkl
from typing import List, Union, Callable
from sklearn.metrics._scorer import _BaseScorer
import numpy as np
import time
import datetime

from mindware.modules.base import BaseAutoML
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode

from mindware.components.config_space.cs_builder import get_cash_cs
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import CONFIGS


class BaseCASH(BaseAutoML):
    
    name='cash'
    
    def __init__(self, include_algorithms: List[str] = None, task_type: int = None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', sub_optimizer: str = 'smac', inner_iter_num_per_iter=1,
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir='./data', seed=1, n_jobs=1, topk=50, rmfiles=False,
                 ensemble_method=None, ensemble_size=5, task_id='test',
                 filter_params=None, optimizer_params=None):

        super(BaseCASH, self).__init__(
            task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, inner_iter_num_per_iter=inner_iter_num_per_iter,
            time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=per_run_time_limit,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs, topk=topk, rmfiles=rmfiles,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size, task_id=task_id, optimizer_params=optimizer_params
        )

        if optimizer not in ['llmmab', 'llmbo', 'smac', 'llm', 'tpe', 'hebo', 'cmaes', 'pt4opt_tpe', 'pt4opt_cmaes', 'lolbo', 'pt4opt_lolbo', 'ptllm', 'ptllm_tpe', 'ptllm_smac', 'ptllm_cmaes', 'ptllm_hebo', 'random_search', 'mab', 'block_0', 'block_1', 'bopro', 'bora', 'opro', 'llambo', 'mosaic']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if sub_optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid sub_optimizer: %s for CASH!' % sub_optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.include_algorithms = include_algorithms
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        path = 'CASH-%s(%d)-%s_%s_%s' % (
            optimizer, self.seed, self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        cs_args = {
            'resampling_params': resampling_params,
            'data_node': data_node
        }
        from mindware.components.config_space.cs_builder import get_cs_args
        cs_args = get_cs_args(**cs_args)
        self.cs_args = cs_args

        self.filter_params = filter_params
        # select models
        if self.filter_params is not None and 'n_algorithm' in self.filter_params:
            n_algo = self.filter_params['n_algorithm']
            include_algorithms = self._recommand_models(self.task_type, task_id=self.task_id, data_node=self.data_node, metric=self.metric_name, n_algo=n_algo, include_algorithms=include_algorithms)

        if include_algorithms is not None and len(include_algorithms) == 1:
            from mindware.components.config_space.cs_builder import get_hpo_cs
            self.cs = get_hpo_cs(estimator_id=include_algorithms[0], task_type=self.task_type, **cs_args)
        else:
            from mindware.components.config_space.cs_builder import get_cash_cs
            self.cs = get_cash_cs(include_algorithms=include_algorithms, task_type=self.task_type, **cs_args)
            
        kd_params = {
            k: v
            for k, v in (self.optimizer_params or {}).items()
            if k not in {
                "task_type",
                "metric",
                "task_id",
                "datanode",
                "config_space",
                "include_algorithms",
                "original_data_dir",
                "rag_data_dir",
            }
        }
        KnowledgeDatabase.instance(
            task_type=task_type, metric=self.metric_name, task_id=task_id, datanode=self.data_node, config_space=self.cs, include_algorithms=include_algorithms,
            original_data_dir=CONFIGS.ORIGINAL_DATA_DIR, rag_data_dir=CONFIGS.RAG_DATA_DIR, **kd_params
        )


        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        # Define evaluator and optimizer

        data_node_path = os.path.join(self.output_dir, 'data_node.pkl')
        with open(data_node_path, 'wb') as f:
            pkl.dump(data_node, f)
        self.evaluator = None
        if self.task_type in CLS_TASKS:
            from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator
            self.evaluator = CASHCLSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node_path,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                if_imbal=self.if_imbal)
        else:
            from mindware.modules.cash.cash_evaluator import CASHRGSEvaluator
            self.evaluator = CASHRGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node_path,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed)

        self.optimizer = self.build_optimizer(name='cash', sub_optimizer=sub_optimizer)

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-CASH-task_type%d-%s(%d)' % (self.task_type, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def get_conf(self, save=False):

        conf = super(BaseCASH, self).get_conf()
        from ConfigSpace.hyperparameters import Constant
        if isinstance(self.cs['algorithm'], Constant):
            conf['include_algorithms'] = [self.cs['algorithm'].value]
        else:
            conf['include_algorithms'] = self.cs['algorithm'].choices

        if save:
            conf_to_save = dict(conf)
            # Live HF clients are not JSON-serializable; persist a path hint instead.
            opt_params = conf_to_save.get("optimizer_params")
            if isinstance(opt_params, dict) and "llm_client" in opt_params:
                opt_params = dict(opt_params)
                client = opt_params.pop("llm_client", None)
                if client is not None and "qwen_adapter_path" not in opt_params:
                    adapter = getattr(client, "adapter_path", None)
                    model_path = getattr(client, "model_path", None)
                    if adapter:
                        opt_params["qwen_adapter_path"] = adapter
                    if model_path:
                        opt_params["qwen_model_path"] = model_path
                conf_to_save["optimizer_params"] = opt_params
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf_to_save, f, indent=4, default=str)

        return conf
import os
import json
import pickle as pkl
from typing import List, Union, Callable
from sklearn.metrics._scorer import _BaseScorer
import numpy as np
import time
import datetime

from mindware.modules.base import BaseAutoML
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode

from mindware.components.config_space.cs_builder import get_cash_cs
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import CONFIGS


class BaseCASH(BaseAutoML):
    
    name='cash'
    
    def __init__(self, include_algorithms: List[str] = None, task_type: int = None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', sub_optimizer: str = 'smac', inner_iter_num_per_iter=1,
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir='./data', seed=1, n_jobs=1, topk=50, rmfiles=False,
                 ensemble_method=None, ensemble_size=5, task_id='test',
                 filter_params=None, optimizer_params=None):

        super(BaseCASH, self).__init__(
            task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, inner_iter_num_per_iter=inner_iter_num_per_iter,
            time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=per_run_time_limit,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs, topk=topk, rmfiles=rmfiles,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size, task_id=task_id, optimizer_params=optimizer_params
        )

        if optimizer not in ['llmmab', 'llmbo', 'smac', 'llm', 'tpe', 'hebo', 'cmaes', 'pt4opt_tpe', 'pt4opt_cmaes', 'lolbo', 'pt4opt_lolbo', 'ptllm', 'ptllm_tpe', 'ptllm_smac', 'ptllm_cmaes', 'ptllm_hebo', 'random_search', 'mab', 'block_0', 'block_1', 'bopro', 'bora', 'opro', 'llambo', 'mosaic']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if sub_optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid sub_optimizer: %s for CASH!' % sub_optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.include_algorithms = include_algorithms
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        path = 'CASH-%s(%d)-%s_%s_%s' % (
            optimizer, self.seed, self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        cs_args = {
            'resampling_params': resampling_params,
            'data_node': data_node
        }
        from mindware.components.config_space.cs_builder import get_cs_args
        cs_args = get_cs_args(**cs_args)
        self.cs_args = cs_args

        self.filter_params = filter_params
        # select models
        if self.filter_params is not None and 'n_algorithm' in self.filter_params:
            n_algo = self.filter_params['n_algorithm']
            include_algorithms = self._recommand_models(self.task_type, task_id=self.task_id, data_node=self.data_node, metric=self.metric_name, n_algo=n_algo, include_algorithms=include_algorithms)

        if include_algorithms is not None and len(include_algorithms) == 1:
            from mindware.components.config_space.cs_builder import get_hpo_cs
            self.cs = get_hpo_cs(estimator_id=include_algorithms[0], task_type=self.task_type, **cs_args)
        else:
            from mindware.components.config_space.cs_builder import get_cash_cs
            self.cs = get_cash_cs(include_algorithms=include_algorithms, task_type=self.task_type, **cs_args)
            
        kd_params = {
            k: v
            for k, v in (self.optimizer_params or {}).items()
            if k not in {
                "task_type",
                "metric",
                "task_id",
                "datanode",
                "config_space",
                "include_algorithms",
                "original_data_dir",
                "rag_data_dir",
            }
        }
        KnowledgeDatabase.instance(
            task_type=task_type, metric=self.metric_name, task_id=task_id, datanode=self.data_node, config_space=self.cs, include_algorithms=include_algorithms,
            original_data_dir=CONFIGS.ORIGINAL_DATA_DIR, rag_data_dir=CONFIGS.RAG_DATA_DIR, **kd_params
        )


        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        # Define evaluator and optimizer

        data_node_path = os.path.join(self.output_dir, 'data_node.pkl')
        with open(data_node_path, 'wb') as f:
            pkl.dump(data_node, f)
        self.evaluator = None
        if self.task_type in CLS_TASKS:
            from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator
            self.evaluator = CASHCLSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node_path,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                if_imbal=self.if_imbal)
        else:
            from mindware.modules.cash.cash_evaluator import CASHRGSEvaluator
            self.evaluator = CASHRGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node_path,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed)

        self.optimizer = self.build_optimizer(name='cash', sub_optimizer=sub_optimizer)

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-CASH-task_type%d-%s(%d)' % (self.task_type, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def get_conf(self, save=False):

        conf = super(BaseCASH, self).get_conf()
        from ConfigSpace.hyperparameters import Constant
        if isinstance(self.cs['algorithm'], Constant):
            conf['include_algorithms'] = [self.cs['algorithm'].value]
        else:
            conf['include_algorithms'] = self.cs['algorithm'].choices

        if save:
            conf_to_save = dict(conf)
            # Live HF clients are not JSON-serializable; persist a path hint instead.
            opt_params = conf_to_save.get("optimizer_params")
            if isinstance(opt_params, dict) and "llm_client" in opt_params:
                opt_params = dict(opt_params)
                client = opt_params.pop("llm_client", None)
                if client is not None and "qwen_adapter_path" not in opt_params:
                    adapter = getattr(client, "adapter_path", None)
                    model_path = getattr(client, "model_path", None)
                    if adapter:
                        opt_params["qwen_adapter_path"] = adapter
                    if model_path:
                        opt_params["qwen_model_path"] = model_path
                conf_to_save["optimizer_params"] = opt_params
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf_to_save, f, indent=4, default=str)

        return conf
import os
import json
import pickle as pkl
from typing import List, Union, Callable
from sklearn.metrics._scorer import _BaseScorer
import numpy as np
import time
import datetime

from mindware.modules.base import BaseAutoML
from mindware.utils.logging_utils import setup_logger, get_logger
from mindware.components.utils.constants import CLS_TASKS
from mindware.components.feature_engineering.transformation_graph import DataNode

from mindware.components.config_space.cs_builder import get_cash_cs
from mindware.knowledge_db.knowledge_database import KnowledgeDatabase
from mindware.modules.configs import CONFIGS


class BaseCASH(BaseAutoML):
    
    name='cash'
    
    def __init__(self, include_algorithms: List[str] = None, task_type: int = None,
                 metric: Union[str, Callable, _BaseScorer] = 'acc', data_node: DataNode = None,
                 evaluation: str = 'holdout', resampling_params=None,
                 optimizer='smac', sub_optimizer: str = 'smac', inner_iter_num_per_iter=1,
                 time_limit=600, amount_of_resource=None, per_run_time_limit=600,
                 output_dir='./data', seed=1, n_jobs=1, topk=50, rmfiles=False,
                 ensemble_method=None, ensemble_size=5, task_id='test',
                 filter_params=None, optimizer_params=None):

        super(BaseCASH, self).__init__(
            task_type=task_type,
            metric=metric, data_node=data_node,
            evaluation=evaluation, resampling_params=resampling_params,
            optimizer=optimizer, inner_iter_num_per_iter=inner_iter_num_per_iter,
            time_limit=time_limit, amount_of_resource=amount_of_resource, per_run_time_limit=per_run_time_limit,
            output_dir=output_dir, seed=seed, n_jobs=n_jobs, topk=topk, rmfiles=rmfiles,
            ensemble_method=ensemble_method, ensemble_size=ensemble_size, task_id=task_id, optimizer_params=optimizer_params
        )

        if optimizer not in ['llmmab', 'llmbo', 'smac', 'llm', 'tpe', 'hebo', 'cmaes', 'pt4opt_tpe', 'pt4opt_cmaes', 'lolbo', 'pt4opt_lolbo', 'ptllm', 'ptllm_tpe', 'ptllm_smac', 'ptllm_cmaes', 'ptllm_hebo', 'random_search', 'mab', 'block_0', 'block_1', 'bopro', 'bora', 'opro', 'llambo', 'mosaic']:
            raise ValueError('Invalid optimizer: %s for CASH!' % optimizer)
        if sub_optimizer not in ['smac', 'tpe', 'random_search']:
            raise ValueError('Invalid sub_optimizer: %s for CASH!' % sub_optimizer)
        if evaluation not in ['holdout', 'cv', 'partial', 'partial_bohb']:
            raise ValueError('Invalid evaluation: %s for CASH!' % evaluation)

        self.include_algorithms = include_algorithms
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        path = 'CASH-%s(%d)-%s_%s_%s' % (
            optimizer, self.seed, self.evaluation, self.task_id, self.datetime
        )
        self.output_dir = os.path.join(output_dir, path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(optimizer)

        cs_args = {
            'resampling_params': resampling_params,
            'data_node': data_node
        }
        from mindware.components.config_space.cs_builder import get_cs_args
        cs_args = get_cs_args(**cs_args)
        self.cs_args = cs_args

        self.filter_params = filter_params
        # select models
        if self.filter_params is not None and 'n_algorithm' in self.filter_params:
            n_algo = self.filter_params['n_algorithm']
            include_algorithms = self._recommand_models(self.task_type, task_id=self.task_id, data_node=self.data_node, metric=self.metric_name, n_algo=n_algo, include_algorithms=include_algorithms)

        if include_algorithms is not None and len(include_algorithms) == 1:
            from mindware.components.config_space.cs_builder import get_hpo_cs
            self.cs = get_hpo_cs(estimator_id=include_algorithms[0], task_type=self.task_type, **cs_args)
        else:
            from mindware.components.config_space.cs_builder import get_cash_cs
            self.cs = get_cash_cs(include_algorithms=include_algorithms, task_type=self.task_type, **cs_args)
            
        kd_params = {
            k: v
            for k, v in (self.optimizer_params or {}).items()
            if k not in {
                "task_type",
                "metric",
                "task_id",
                "datanode",
                "config_space",
                "include_algorithms",
                "original_data_dir",
                "rag_data_dir",
            }
        }
        KnowledgeDatabase.instance(
            task_type=task_type, metric=self.metric_name, task_id=task_id, datanode=self.data_node, config_space=self.cs, include_algorithms=include_algorithms,
            original_data_dir=CONFIGS.ORIGINAL_DATA_DIR, rag_data_dir=CONFIGS.RAG_DATA_DIR, **kd_params
        )


        self.timestamp = time.time()
        self.datetime = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d-%H-%M-%S-%f')
        # Define evaluator and optimizer

        data_node_path = os.path.join(self.output_dir, 'data_node.pkl')
        with open(data_node_path, 'wb') as f:
            pkl.dump(data_node, f)
        self.evaluator = None
        if self.task_type in CLS_TASKS:
            from mindware.modules.cash.cash_evaluator import CASHCLSEvaluator
            self.evaluator = CASHCLSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node_path,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed,
                if_imbal=self.if_imbal)
        else:
            from mindware.modules.cash.cash_evaluator import CASHRGSEvaluator
            self.evaluator = CASHRGSEvaluator(
                fixed_config=None,
                scorer=self.metric,
                data_node=data_node_path,
                resampling_strategy=self.evaluation,
                resampling_params=self.resampling_params,
                timestamp=self.timestamp,
                output_dir=self.output_dir,
                seed=self.seed)

        self.optimizer = self.build_optimizer(name='cash', sub_optimizer=sub_optimizer)

    def _get_logger(self, optimizer_name):
        logger_name = 'MindWare-CASH-task_type%d-%s(%d)' % (self.task_type, optimizer_name, self.seed)
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def get_conf(self, save=False):

        conf = super(BaseCASH, self).get_conf()
        from ConfigSpace.hyperparameters import Constant
        if isinstance(self.cs['algorithm'], Constant):
            conf['include_algorithms'] = [self.cs['algorithm'].value]
        else:
            conf['include_algorithms'] = self.cs['algorithm'].choices

        if save:
            conf_to_save = dict(conf)
            # Live HF clients are not JSON-serializable; persist a path hint instead.
            opt_params = conf_to_save.get("optimizer_params")
            if isinstance(opt_params, dict) and "llm_client" in opt_params:
                opt_params = dict(opt_params)
                client = opt_params.pop("llm_client", None)
                if client is not None and "qwen_adapter_path" not in opt_params:
                    adapter = getattr(client, "adapter_path", None)
                    model_path = getattr(client, "model_path", None)
                    if adapter:
                        opt_params["qwen_adapter_path"] = adapter
                    if model_path:
                        opt_params["qwen_model_path"] = model_path
                conf_to_save["optimizer_params"] = opt_params
            with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
                json.dump(conf_to_save, f, indent=4, default=str)

        return conf
