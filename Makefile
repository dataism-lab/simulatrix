SHELL=/bin/bash

##########
# Defaults
##########

ROOT=$(shell pwd)

VENV_DIR=${ROOT}/venv
PY_EXEC=${VENV_DIR}/bin/python
PY_PATH=$(shell find ${VENV_DIR} -name site-packages):${ROOT}/src
python=PYTHONPATH=${PY_PATH} ${PY_EXEC}

#-- GPU parametrisation --
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=True

#-- Parameters --
HOST=127.0.0.1
IMG_NAME=docetti/carla:0.9.13-1
MAPNAME=/Game/map_package/Maps/expedition_loop_bordered/expedition_loop_bordered

######
# Demo
######

demo_bew:
	$(python) -m tools.bew.demo_bew --host ${HOST}

demo_gym:
	$(python) -m tools.gymcustom.demo_env --host ${HOST}

#####################
# Simulator container
#####################

#-- off-screen --
sim_run:
	cd carlasim && \
	docker compose up --force-recreate --always-recreate-deps --remove-orphans -d

sim_stop:
	cd carlasim && \
	docker compose down

#-- on-screen --
sim_render:
	cd carlasim && \
	docker compose -f docker-compose.render.yml up --force-recreate --always-recreate-deps --remove-orphans

sim_render_stop:
	cd carlasim && \
	docker compose -f docker-compose.render.yml down

#######
# UTILS
#######

stop_carla:
	docker stop $$(docker ps -aqf ancestor="${IMG_NAME}")

stop_containers:
	docker stop $$(docker ps -a -q) || true
	docker rm $$(docker ps -a -q) || true

clean_dangling: stop_containers
	docker rmi $(shell docker images --filter "dangling=true" -q --no-trunc)

ls_port:
	sudo lsof -nP -i | grep LISTEN