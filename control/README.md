# CarlaGym-DeepRacer Training

## Overview
1. The configuration file: `resources/configs/ppo_camera_config.yaml` or `resources/configs/ppo_vlidar_config.yaml`
2. The training file: ppo_train.py
3. The inference file: ppo_inference_ray.py 

RLLib Core concepts [Link](https://docs.ray.io/en/latest/rllib/core-concepts.html)

## Local train

Before training the policy, you should modify `./control/Makefile`
- set or remove `--mlflow_uri` with tracking url;
- enable or disable `--state_log` for saving rollout ego states that are passed to reward function in every step for further analysis;
- set `--stop_iteration` (ex 30 training iterations) or `--stop_duration` (ex. 360s, 10m, 1h).
- set `NAME` variable in Makefile (name of the experiment that would became the final folder name).
- set `CONFIG` variable in Makefile that's a full path to one of the config file from `./resoures/control_config` folder.
```bash
cd ./control
make train
# or run with mlflow enabled
make train_mlflow
```

Continue train from last checkpoint
```bash
cd ./control
make train_restore 
```

Render rollout with the selected checkpoint.
- set `CHECKPOINT` variable in Makefile with full path to a checkpoint folder.  
```bash
cd ./control
make rollout
```

You could freely modify configs in `./resoures/control_config/*.yaml`.
For ease of experiment you could set `avoid_server_init` to `True` in order to use an existing carla server's endpoint otherwise
it'll create a new carla server container.

### Training notes
You could enable evalution by setting a number in `evaluation_interval` in training config file.

If you enable evaluation you should know that while training, there'll be created a standalone carla server
for evaluation purpose, since it's drawbacks of RLLib that we currently can't resolve easily. 

## PPO parameters
- **min_sample_timesteps_per_iteration** (default = 20000) - after one `step_attempt()`, the timestep counts (sampling or
training) have not been reached, will perform n more `step_attempt()`
calls until the minimum timesteps have been executed.
- **learning_starts** (default = 20000) - How many steps of the model to sample before learning starts.
- **buffer_size** (default = 5000) - Size of the replay buffer. Note that if async_updates is set, then
  each worker will have a replay buffer of this size.
- **clip_rewards** (default = None) -
  - True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0; 
  - False: Never clip;
  - [float value]: Clip at -value and + value; 
  - Tuple[value1, value2]: Clip at value1 and value2.
- **clip_param** (default = 0.4) - PPO surrogate loss options.

## Distributed training params
Documentation [Link](https://docs.ray.io/en/master/rllib/rllib-sample-collection.html)

- **rollout_fragment_length** (default = ) -
  - LSTM state is not backpropagated across rollout fragments. 
  - Value function is used to bootstrap rewards past the end of the sequence. 
  - Affects system performance (too small => high overhead, too large => high delay).
- **num_sgd_iter** (default = 30) - Number of SGD iterations in each outer loop.
The minibatch ring buffer stores and replays batches of size `train_batch_size`
up to `num_sgd_iter` times per batch. Number of epochs to
execute per train batch
- **sgd_minibatch_size** (default = 128) - is the size that gets sampled from the training buffer
- **train_batch_size** (default = 4000) - number of timesteps collected for each SGD round. This defines the size
of each SGD epoch.
- **batch_mode** (default = truncate_episodes) - truncate_episodes, complete_episodes
- **observation_filter** (default = NoFilter) - NoFilter, MeanStdFilter
- **vf_share_layers** (default = True) - whether layers should be shared for the value function.

### In general
**sgd_minibatch_size**: PPO takes a train batch (of size train_batch_size)
and chunks it down into n *sgd_minibatch_size* sized pieces.
E.g. if train_batch_size=1000 and sgd_minibatch_size=100,
then we create 10 “sub-sampling” pieces out of the train batch.
*num_sgd_iter*: The above sub-sampling pieces are then fed
*num_sgd_iter* times to the NN for updating. 

### PPO specific settings:
     self.lr_schedule = None
     self.use_critic = True
     self.use_gae = True
     self.lambda_ = 1.0
     self.kl_coeff = 0.2
     self.sgd_minibatch_size = 128
     self.num_sgd_iter = 30
     self.shuffle_sequences = True
     self.vf_loss_coeff = 1.0
     self.entropy_coeff = 0.0
     self.entropy_coeff_schedule = None
     self.clip_param = 0.3
     self.vf_clip_param = 10.0
     self.grad_clip = None
     self.kl_target = 0.01
   
     # Override some of AlgorithmConfig's default values with PPO-specific values.
     self.rollout_fragment_length = 200
     self.train_batch_size = 4000
     self.lr = 5e-5
     self.model["vf_share_layers"] = False

## Config overview
General config overview [Link](https://docs.ray.io/en/latest/rllib/rllib-training.html)

Model specific config overview [Link](https://docs.ray.io/en/latest/rllib/rllib-models.html)

## Mlflow logging
For experiment tracking we developed a `./control/intergration/callbacks/CustomMlflowCallback`.
There we fix a problem of varchar-256 limitation for parameters saving
from the given config.  