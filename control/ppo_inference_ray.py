from __future__ import print_function

import argparse

import ray
import yaml
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from tools.gymcustom.gym_carla import env_factory, kill_all_servers


def parse_config(args):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["num_workers"] = 0
        config["explore"] = False
        config['num_envs_per_worker'] = 1
        config['env_config']['carla']['avoid_server_init'] = True
        del config["num_cpus_per_worker"]
        del config["num_gpus_per_worker"]

    return config


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint from which to roll out.")

    args = argparser.parse_args()
    args.config = parse_config(args)

    env_name = args.config['env']
    register_env(env_name, lambda config: env_factory(config, env_type=env_name))

    try:
        info = ray.init(address=None, ignore_reinit_error=False, log_to_driver=True, include_dashboard=False)
        print(info)

        # Restore agent
        agent = PPOTrainer(config=args.config)
        agent.restore(args.checkpoint)

        # Initalize the CARLA environment
        env = agent.workers.local_worker().env
        obs = env.reset()

        while True:
            action = agent.compute_action(obs)
            obs, _, done, _ = env.step(action)
            env.render()

            if done:
                print('steps', env.time_step, 'reset_count', env.reset_step, 'speed', obs[-2])
                obs = env.reset()

    except KeyboardInterrupt:
        print("\nshutdown by user")
    finally:
        ray.shutdown()
        kill_all_servers()


if __name__ == "__main__":
    main()
