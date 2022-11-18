from __future__ import print_function

import argparse
import os
from typing import Dict

import ray
import yaml
from ray import tune
from ray.rllib.algorithms.callbacks import MultiCallbacks
from ray.tune.registry import register_env

from control.integration.callbacks import CustomMlflowCallback, StateLoggerCallbacks, ProgressCallback
from control.integration.helper import TimeStopper
from control.integration.helper import get_checkpoint
from control.ppo_ray.ppo_trainer import CustomPPOTrainer
from tools.common.exceptions import InvalidConfigException
from tools.gymcustom.gym_carla import env_factory, kill_all_servers


def run(args):
    env_name = args.config['env']
    register_env(env_name, lambda config: env_factory(config, env_type=env_name))
    try:
        info = ray.init(address="auto" if args.auto else None,
                        ignore_reinit_error=False, log_to_driver=True, include_dashboard=False)
        print(info)
        # print(CustomPPOTrainer.default_resource_request(config=args.config)._bundles)
        if args.debug:
            trainer = CustomPPOTrainer(config=args.config)
            while True:
                print(trainer.train())
        else:
            store_directory = get_checkpoint(args.name, args.directory, args.restore, args.overwrite)

            tune_callbacks = []
            if args.mlflow_uri:
                s3_confs = {
                    'AWS_ACCESS_KEY_ID': os.environ['AWS_ACCESS_KEY_ID'],
                    'AWS_SECRET_ACCESS_KEY': os.environ['AWS_SECRET_ACCESS_KEY'],
                    'MLFLOW_S3_ENDPOINT_URL': os.environ['MLFLOW_S3_ENDPOINT_URL']
                }
                tune_callbacks.append(CustomMlflowCallback(
                    tracking_uri=args.mlflow_uri,
                    registry_uri=args.mlflow_uri,
                    experiment_name=args.name,
                    save_artifact=True,
                    s3_confs=s3_confs,
                    tags=None)
                )
            stop_criteria = None
            if args.stop_duration:
                stop_criteria = TimeStopper(args.duration)
            elif args.stop_iteration:
                stop_criteria = {"training_iteration": args.stop_iteration}
                # "perf/ram_util_percent": 90.0

            checkpoint_freq = args.config.get('evaluation_interval', 1)
            if not checkpoint_freq:
                checkpoint_freq = 1

            # add custom metrics if needed
            # reporter.add_metric_column("")
            tune.run(CustomPPOTrainer,
                     name=args.name,

                     keep_checkpoints_num=2,
                     checkpoint_score_attr='episode_reward_mean',
                     checkpoint_freq=checkpoint_freq,
                     checkpoint_at_end=True,

                     restore=store_directory,
                     config=args.config,
                     verbose=False,
                     reuse_actors=True,
                     local_dir=args.directory,
                     stop=stop_criteria,
                     callbacks=tune_callbacks
                     )

    finally:
        kill_all_servers()
        ray.shutdown()


def parse_config(args) -> Dict:
    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if 'env' not in config:
            raise InvalidConfigException('env is empty')
    # particularly check config
    assert config['train_batch_size'] % config['rollout_fragment_length'] == 0, 'Incompatible rollout'
    assert config['train_batch_size'] % config['num_envs_per_worker'] == 0, 'Hard to split target count per actor'

    # enable custom callbacks
    callbacks = [ProgressCallback]
    if args.state_log:
        callbacks.append(StateLoggerCallbacks)

    if callbacks:
        config['callbacks'] = MultiCallbacks(callbacks)

    return config


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument("-d", "--directory", metavar='D',
                           default=os.path.expanduser("~") + "/ray_results/carla_rllib",
                           help="Specified directory to save results (default: ~/ray_results/carla_rllib")
    argparser.add_argument("-n", "--name", metavar="N", default="ppo_example",
                           help="Name of the experiment (default: ppo_example)")
    argparser.add_argument("--restore", action="store_true", default=False,
                           help="Flag to restore from the specified directory")
    argparser.add_argument("--overwrite", action="store_true", default=False,
                           help="Overwrite a specific directory (warning: all content of the folder will be lost.)")
    argparser.add_argument("--auto", action="store_true", default=False,
                           help="Flag to use auto address")
    argparser.add_argument("--state_log", action="store_true", default=False,
                           help="Enable state logging while training an agent")
    argparser.add_argument("--mlflow_uri", default=None, type=str,
                           help='tracking url to an open mlflow endpoint')
    argparser.add_argument("--stop_duration", default=None, type=str,
                           help='Duration for training ex. 100s, 4m, 2h for seconds, minutes and hours respectively,'
                                'stop_iteration should be void')
    argparser.add_argument("--stop_iteration", default=None, type=int,
                           help='Total count of training iteration before stop, stop_duration should be void')
    argparser.add_argument('--debug', default=False, action='store_false',
                           help='Enable debug in PyCharm')

    args = argparser.parse_args()
    args.config = parse_config(args)

    run(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
