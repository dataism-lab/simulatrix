# 2022: Ildar Nurgaliev (i.nurgaliev@docet.ai)

import copy
import logging
import os
from typing import Dict, Optional

from ray.air.callbacks.mlflow import MLflowLoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.offline.io_context import IOContext
from ray.tune import Callback

from control.integration.writer import AsyncJsonWriter

logger = logging.getLogger(__name__)


class TerminalInfoCallback(Callback):
    """
    Just log in terminal some basic metrics
    """

    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result: Mean reward={result['episode_reward_mean']:.2f}; "
              f"Mean episode={result['episode_len_mean']:.2f}")


class ProgressCallback(DefaultCallbacks):

    def on_train_result(self, *, algorithm=None, result: Dict, trainer=None, **kwargs) -> None:
        print('training_iteration:', result['training_iteration'],
              'with evaluation' if 'evaluation' in result else '')


class StateLoggerCallbacks(DefaultCallbacks):
    """
    Collect Core-state per trajectories
    """

    # Only one worker would write down all the training samples
    WORKER_INDEX = 0

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self._training_iteration = 0
        self._eval_iteration = 0
        self._writer = None
        self._is_train = True
        self.log_dir = None
        self._batch_size = 0
        self._prev_unroll_id = 0
        self.evaluation_interval = 0

    def _setup_writer(self) -> None:
        assert self.log_dir, 'log_dir is empty'
        if self._is_train:
            prefix = f"train-{self._training_iteration:02d}"
        else:
            prefix = 'eval'
        if self._writer:
            self._writer.close()
        self._writer = AsyncJsonWriter(self.log_dir, ioctx=IOContext(worker_index=self.WORKER_INDEX),
                                       prefix=prefix,
                                       avoid_columns=frozenset(['obs', 'new_obs', 'infos', 'agent_index']))

    def on_sub_environment_created(self, *, worker, sub_environment, env_context, **kwargs) -> None:
        if self._writer or worker.worker_index != self.WORKER_INDEX:
            return

        cfg = worker.creation_args()
        is_eval: bool = cfg['policy_config']['in_evaluation']
        self.log_dir = os.path.join(cfg['log_dir'], "rollouts")
        self.evaluation_interval = cfg['policy_config']['evaluation_interval']
        if is_eval:
            self._is_train = False
        else:
            self._batch_size = cfg['policy_config']['train_batch_size']
            print(f'log_dir: {self.log_dir}')
            assert self._batch_size > 0

        worker.env.enable_state_info()
        self._setup_writer()

    def on_sample_end(self, *, worker, samples: "SampleBatch", **kwargs) -> None:
        """
        Called at the end of RolloutWorker.sample()
        """
        # print('global_timestep', worker.get_policy().global_timestep)
        # print('sampels-len', len(samples), 'on rollout_length', worker.rollout_fragment_length, 'training_iteration', self._training_iteration)
        if worker.worker_index != self.WORKER_INDEX:
            return

        if self._is_train:
            if worker.get_policy().global_timestep % self._batch_size == 0:
                self._training_iteration += 1
                self._setup_writer()
        else:
            unroll_id: int = samples['unroll_id'][0]
            if unroll_id - self._prev_unroll_id > 1:
                self._training_iteration += self.evaluation_interval
            self._prev_unroll_id = unroll_id

        extra_info = {'train_iteration': [self._training_iteration] * len(samples)}
        self._writer.write(samples, extra_info=extra_info)

    def on_train_result(self, *, algorithm=None, result: Dict, trainer=None, **kwargs) -> None:
        print('training_iteration:', result['training_iteration'], 'with evaluation' if 'evaluation' in result else '')


class CustomMlflowCallback(MLflowLoggerCallback):
    """
    Avoid problem with the "varchar(256) is too short"
    """

    INCLUDE_RESULT = {'time_this_iter_s', 'training_iteration'}

    def __init__(self, tracking_uri: Optional[str], registry_uri: Optional[str],
                 experiment_name: Optional[str], save_artifact: bool,
                 s3_confs: Dict[str, str], tags: Optional[Dict] = None):
        assert not save_artifact or all(s3_confs.values()), "s3_confs has an empty definition," \
                                                            "it should be defined for every key"

        for k, v in s3_confs.items():
            os.environ[k] = v

        super().__init__(tracking_uri, registry_uri, experiment_name, tags, save_artifact)

    def _filter_result(self, result: Dict, prefix='') -> Dict:
        return {prefix + k: v for k, v in result.items() if k in self.INCLUDE_RESULT or
                any(k.endswith(postfix) for postfix in ['_min', '_max', '_mean'])}

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        result_filtered = self._filter_result(result)
        if 'evaluation' in result:
            result_filtered.update(self._filter_result(result['evaluation'], prefix='eval_'))
            if 'custom_metrics' in result['evaluation']:
                result_filtered.update(self._filter_result(result['evaluation']['custom_metrics'], prefix='eval_'))
        if 'custom_metrics' in result:
            result_filtered.update(self._filter_result(result['custom_metrics']))

        super().log_trial_result(iteration, trial, result_filtered)

    def log_trial_start(self, trial: "Trial"):
        config = trial.config

        if 'env_config' in config:
            config_report = copy.deepcopy(config)
            del config_report['env_config']
            for k, v in config['env_config'].items():
                if type(v) == dict:
                    for k2, v2 in v.items():
                        config_report[f'{k}_{k2}'] = v2
                else:
                    config_report[k] = v

            trial.config = config_report
            super().log_trial_start(trial)
            trial.config = config
        else:
            super().log_trial_start(trial)
