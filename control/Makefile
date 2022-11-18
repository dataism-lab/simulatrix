SHELL=/bin/bash

##########
# Defaults
##########

ROOT=$(shell dirname $$(pwd))

VENV_DIR=${ROOT}/venv
PY_EXEC=${VENV_DIR}/bin/python
PY_PATH=$(shell find ${VENV_DIR} -name site-packages):${ROOT}
python=PYTHONPATH=${PY_PATH} ${PY_EXEC}

#-- GPU parametrisation --
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=True

#-- mlflow --
# export variables without quotes
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:19000
MLFLOW_TRACKING_URI="http://localhost:5000"

#-- Parameters --
HOST=127.0.0.1
#CONFIG=${ROOT}/resources/control_configs/ppo_vlidar_config.yaml
CONFIG=${ROOT}/resources/control_configs/ppo_camera_config.yaml

#-- experiment dir of already trained model --
NAME="ppo_example"
EX_DIR=~/ray_results/carla_rllib/ppo_example/CustomPPOTrainer_carla-racer-minmax_edce1_00000_0_2022-10-19_22-55-06
CHECKPOINT=${EX_DIR}/checkpoint_000004

#######
# Train
#######

train:
	$(python) ppo_train.py ${CONFIG} -n ${NAME} --overwrite --stop_iteration 10 --state_log

train_mlflow:
	$(python) ppo_train.py ${CONFIG} -n ${NAME} --overwrite --stop_iteration 10 \
						--mlflow_uri ${MLFLOW_TRACKING_URI} --state_log


train_restore:
	$(python) ppo_train.py ${CONFIG} --restore --stop_duration 1h \
						--mlflow_uri ${MLFLOW_TRACKING_URI} --state_log

#-- Watch trained model driving a car --
rollout:
	$(python) ppo_inference_ray.py ${CONFIG} -n ${NAME}  --checkpoint "${CHECKPOINT}"

#######
# UTILS
#######

last:
	ls -1Rt ${CUR_DIR} | head -n 3

board:
	source ${VENV_DIR}/bin/activate && \
	tensorboard --logdir ${CUR_DIR}
