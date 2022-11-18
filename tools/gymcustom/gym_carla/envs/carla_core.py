# 2022: Ildar Nurgaliev (i.nurgaliev@docet.ai)

import logging
import os
import pickle
import random
import subprocess
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Dict, List

import carla
import numpy as np
import psutil

from tools.common import utils
from tools.gymcustom.gym_carla.envs import misc
from tools.gymcustom.gym_carla.envs.spawn_controller import SpawnController
from tools.gymcustom.gym_carla.envs.weather import Weather

_root_fld = Path(__file__).absolute().parent.parent.parent.parent.parent.as_posix()
logger = logging.getLogger(__name__)

BASE_CORE_CONFIG = dict(
    host="127.0.0.1",  # Client host
    timeout=10.0,  # Timeout of the client
    timestep=1. / 25,  # time interval between two frames or 1./FPS
    retries_on_error=10,  # Number of tries to connect to the client
    enable_map_assets=False,  # enable / disable all town assets except for the road
    enable_rendering=True,  # enable / disable camera images
    avoid_server_init=True,  # Create or use an existing server
    port=2000,  # default port, used in case of standalone server
    quality_level="Low"  # [Low, Epic]
)

BASE_EXP_CONFIG = dict(
    track='/Game/map_package/Maps/expedition_loop_bordered/expedition_loop_bordered',  # which town to simulate
    ego_vehicle_filter='vehicle.kart.kart',  # filter for defining ego vehicle
    spawn_points=[],
    spawn_mode='random',  # [random, consequent, TODO round-robin]
    n_spawn_points=30,
    consequent_mode_step=1,
    spawn_off_center=False,
    background=dict(
        number_of_vehicles=0,
        weather='ClearNoon',
        other_vehicle_filter='vehicle.kart.*',

    ),
    augmentations=dict(
        throttle_noise_std=0.1,
        brake_noise_std=0.1,
        steering_noise_std=0.01,
        randomize_fps=True,
        randomize_fps_every=60,
        low_friction_enable=False,
        low_friction_n=1,
        low_friction_multiplier=0.7,
        low_friction_areas_size=100,
        weather_change_enable=False,
        weather_update_freq=10,
        weather_speed_factor=1,
    ),
)

CARLA_COMPOSE_FLD = os.path.join(_root_fld, 'carlasim')
IMAGE_NAME = os.environ.get('CARLA_IMG', 'docetti/carla:0.9.13-1')


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def kill_all_servers():
    # cmds = [f'cd {CARLA_COMPOSE_FLD}',
    #         '&&',
    #         'docker compose down']

    cmds = ["docker stop ",
            f'$(docker ps -aqf ancestor="{IMAGE_NAME}")']

    cmd_str = " ".join(map(str, cmds))
    _ = subprocess.Popen(
        cmd_str,
        shell=True,
        preexec_fn=os.setsid,
        stdout=open(os.devnull, "w"),
    )


# special container to avoid warnings and memory leaks. Don't remove!!!
global_sensor_list = []


class CarlaCore:
    """
    Class responsible for handling init Carla service, server-client connecting.
    """

    def __init__(self, core_config: Dict, exp_config: Dict):
        """Initialize the server and client"""
        self.client = None
        self.world = None
        self.map = None
        self.config = utils.join_dicts(BASE_CORE_CONFIG, core_config)
        self.server_port = core_config['port']
        if not core_config['avoid_server_init']:
            self.init_server()
        self.settings = None
        self._container_id: str = ""
        self.timestep = self.config['timestep']
        self.connect_client()

        exp_config = utils.join_dicts(BASE_EXP_CONFIG, exp_config)
        self.exp_config = exp_config
        self._setup_experiment(exp_config)
        self._vehicle_polygons = []

        self._max_waypt = 12  # maximum number of waypoints for hazard detector
        self.max_ego_spawn_times = 10  # maximum times to spawn ego vehicle
        self.max_past_step = 1  # the number of past steps to imitate polygon

        aug_conf = exp_config['augmentations']
        self.randomize_fps = aug_conf['randomize_fps']

        if self.timestep > 0 and not self.randomize_fps:
            # TODO: this has to be reevaluated every time the timestep (fps) is changed
            self._velocity_stack_size = int(5.5 * (1 / self.timestep))
        else:
            self._velocity_stack_size = 90
        self._velocity_queue = deque([1] * self._velocity_stack_size)

        self.throttle_std = aug_conf['throttle_noise_std']
        self.brake_std = aug_conf['brake_noise_std']
        self.steer_std = aug_conf['throttle_noise_std']
        self._past_steering = 0.

        self.randomize_fps_every = aug_conf['randomize_fps_every']
        self._till_new_fps_value = 0

        if aug_conf['weather_change_enable']:
            self.weather = Weather(self.world.get_weather(), aug_conf['weather_update_freq'],
                                   aug_conf['weather_speed_factor'])
        else:
            self.weather = None

        self._low_friction_enable = aug_conf['low_friction_enable'] and aug_conf['low_friction_n'] > 1

    def init_server(self):
        """Start a server on a random port"""
        self.server_port = random.randint(15000, 32000)
        # self.server_port = 2000

        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print(f"Is using the server port: {self.server_port}")
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port + 1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port + 1)

        os.environ['CARLA_RPC_PORT'] = str(self.server_port)
        os.environ['CARLA_STREAM_PORT'] = str(self.server_port + 2)

        # server_command = [f"cd {CARLA_COMPOSE_FLD}",
        #                   '&&',
        #                   'docker compose up',
        #                   '--force-recreate --always-recreate-deps --remove-orphans -d']
        server_command = [
            "docker run ",
            "-d --rm --gpus all "
            f"-p {self.server_port}-{self.server_port + 2}:{self.server_port}-{self.server_port + 2} ",
            f"-v {CARLA_COMPOSE_FLD}/NoEditorSim/DefaultEngine.ini:/home/carla/CarlaUE4/Config/DefaultEngine.ini:ro "
            f"{IMAGE_NAME} ",
            f"/bin/bash ./CarlaUE4.sh -carla-rpc-port={self.server_port} ",
            f"-RenderOffScreen -quality-level=${self.config['quality_level']} -nosound -carla-no-hud"
        ]

        server_command_text = " ".join(map(str, server_command))
        logger.debug(server_command_text)
        out = subprocess.check_output(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid
        )
        assert out, f"An empty output after start container. Out: '{out}'"
        self._container_id = out.strip().decode()

    def stop_server(self):
        if not self._container_id:
            logger.info(f"container_id is empty, thus nothing to stop")
            return
        server_command = ["docker stop ",
                          self._container_id]
        server_command_text = " ".join(map(str, server_command))
        logger.debug(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        """Connect to the client"""

        for i in range(self.config["retries_on_error"]):
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                logger.debug('Got connection with server, check client')
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()
                self.map = self.world.get_map()
                self.settings = self.world.get_settings()
                self.settings.no_rendering_mode = not self.config["enable_rendering"]
                self.set_synchronous_mode(synchronous=True)
                logger.info('Carla server connected!')
                return

            except Exception as e:
                traceback.print_exc()
                logger.info(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1, self.config[
                    "retries_on_error"]))
                time.sleep(3)

        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    def set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode."""
        if synchronous:
            self.settings.fixed_delta_seconds = self.timestep
            self.settings.synchronous_mode = True
            if self.timestep > 0:
                self.settings.substepping = True
                # max_substep_delta_time should be below 0.01
                self.settings.max_substep_delta_time = 0.01
                # MUST fixed_delta_seconds <= max_substep_delta_time * max_substeps
                self.settings.max_substeps = max(10, int(self.timestep / 0.01))
        else:
            self.settings.fixed_delta_seconds = None
            self.settings.synchronous_mode = False

        self.world.apply_settings(self.settings)
        self.world.tick()

    def _load_waypoints(self):
        short_track_name = self.track_name.rsplit('/')[-1]
        store_fpath = utils.get_abs_resourcepath('waypoints')
        pickle_fpath = os.path.join(store_fpath, short_track_name, f'{short_track_name}-final.pickle')

        with open(pickle_fpath, 'rb') as f:
            way_d = pickle.load(f)

        self.waypoints = way_d['center']
        self._inbound_way = way_d['inbound']
        self._outbound_way = way_d['outbound']
        self._track_width = np.linalg.norm(self._outbound_way - self._inbound_way, axis=1)
        self._track_length = misc.track_length(self.waypoints)

    def _setup_experiment(self, exp_config: Dict) -> None:
        """Initialize the hero and sensors"""

        # 1. Sync expected track
        self.track_name = exp_config['track']
        if not self.track_name.endswith(self.world.get_map().name):
            self.world = self.client.load_world(
                map_name=self.track_name,
                reset_settings=False,
                map_layers=carla.MapLayer.All if self.config["enable_map_assets"] else carla.MapLayer.NONE)

        # 2. Load waypoints
        self._load_waypoints()
        # 3. create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(exp_config['ego_vehicle_filter'], color='49,8,8')
        # 4. save info about spawning
        self.spawn_controller = SpawnController(
            exp_config, self.waypoints, self._inbound_way, seed=42)
        # 5. set-up background activities
        back_config = exp_config['background']
        # 5.1 weather
        weather = getattr(carla.WeatherParameters, back_config["weather"])
        self.world.set_weather(weather)
        # 5.2 other vehicles
        self.number_of_vehicles = int(back_config['number_of_vehicles'])
        self.other_vehicle_filter = back_config['other_vehicle_filter']

    def apply_action(self, acc: float, steer: float) -> None:
        # Convert acceleration to throttle and brake
        throttle, brake = self.acc_to_throttle_break(acc)

        # Add noise
        throttle, brake, steer = self._augment_actions(throttle, brake, steer)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if steer > self._past_steering + 0.1:
            steer = self._past_steering + 0.1
        elif steer < self._past_steering - 0.1:
            steer = self._past_steering - 0.1

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle),
                                   steer=float(-steer),
                                   brake=float(brake),
                                   hand_brake=False,
                                   manual_gear_shift=False,
                                   reverse=False)

        self.ego.apply_control(act)
        # flush the action computed with current FPS
        self.world.tick()

        # update weather
        if self.weather is not None:
            self.weather.tick(self.timestep, self.world)

        self._till_new_fps_value -= 1
        if self.randomize_fps and self._till_new_fps_value <= 0:
            # Set fps to a value in range 20 to 40
            self.timestep = np.random.uniform(0.025, 0.05, 1).item()

            self._till_new_fps_value = self.randomize_fps_every
            # Apply new settings and make the world tick
            self.set_synchronous_mode(True)

        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self._vehicle_polygons.append(vehicle_poly_dict)
        while len(self._vehicle_polygons) > self.max_past_step:
            self._vehicle_polygons.pop(0)

    def _augment_actions(self, throttle, brake, steer) -> (float, float, float):
        if throttle != 0:
            # pedals don't push themselves
            throttle += np.random.normal(0, self.throttle_std, 1)
            if throttle < 0:
                throttle = 0
        if brake != 0:
            brake += np.random.normal(0, self.brake_std, 1)
            if brake < 0:
                brake = 0
        steer += np.random.normal(0, self.steer_std, 1)
        return throttle, brake, steer

    def reset(self) -> None:
        # Delete sensors, vehicles and walkers
        self.clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb',
                               'vehicle.*',
                               'controller.ai.walker', 'walker.*',
                               'static.trigger.friction'])
        # Disable sync mode for smooth reset process
        self.set_synchronous_mode(False)

        # Spawn surrounding vehicles
        if self.number_of_vehicles > 0:
            spawn_indices = np.random.permutation(range(len(self.spawn_controller.spawn_transforms)))
            count = self.number_of_vehicles
            if count > 0:
                for spawn_index in spawn_indices:
                    if self._try_spawn_random_vehicle_at(
                            self.spawn_controller.spawn_transforms[spawn_index],
                            number_of_wheels=[4]
                    ):
                        count -= 1
                    if count <= 0:
                        break
            while count > 0:
                if self._try_spawn_random_vehicle_at(
                        random.choice(self.spawn_controller.spawn_transforms),
                        number_of_wheels=[4]
                ):
                    count -= 1
        if self._low_friction_enable:
            self.add_low_traction_areas()

        # Get actors polygon list
        # we need this to avoid spawn over an existing vehicle or pedestrian
        self._vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self._vehicle_polygons.append(vehicle_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            # in case of problems with spawn process
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()

            transform = self.spawn_controller.get_spawn_transform()

            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)

        # reset stuck detector
        self._velocity_queue = deque([1] * self._velocity_stack_size)
        self._till_new_fps_value = self.randomize_fps_every

        self._past_steering = 0

    def _get_nearest_waypoint_idx(self, ego_x, ego_y) -> int:
        deltas = self.waypoints - [ego_x, ego_y]
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2).item()

    def get_state(self) -> Dict:
        """deepracer style state retrieve, all angles are in degree"""
        # ego features
        v = self.ego.get_velocity()
        speed_vec = [v.x, v.y]
        speed = np.linalg.norm(speed_vec)
        speed_longitudinal = misc.get_vehicle_lon_speed(self.ego).item()
        steering = self.ego.get_control().steer
        self._past_steering = steering
        ego_x, ego_y, yaw = misc.get_pos(self.ego)
        ego_point = np.array([ego_x, ego_y])

        # waypoint related calculations
        way_count = self.waypoints.shape[0]
        way_i = self._get_nearest_waypoint_idx(ego_x, ego_y)
        next_way_i = (way_i + 1) % way_count
        prev_way_i = (way_i - 1) if way_i > 0 else (way_count - 1)
        cur_point = self.waypoints[way_i]
        prev_point = self.waypoints[prev_way_i]
        next_point = self.waypoints[next_way_i]
        way_orientation_angle = misc.get_way_direction(cur_point, next_point)
        way_orientation_vec = misc.angle_vec(way_orientation_angle)
        track_width = self._track_width[way_i]

        # ego-waypoint features calculation
        dis = misc.get_lane_dis(prev_point, cur_point, ego_point)

        # longitudinal speed wrt the closest waypoint orientation
        speed_lon_waypoint = np.dot(speed_vec, way_orientation_vec)

        delta_yaw, is_reversed = misc.angle_difference(way_orientation_angle, yaw)

        ego_polygon: np.ndarray = misc.get_bounding_polygon(self.ego)
        is_left_of_center = misc.is_left_of_center(prev_point, cur_point, ego_point)
        poly_dis = misc.get_lane_dis(prev_point, cur_point, ego_polygon)

        all_wheels_on_track = (poly_dis <= track_width / 2.).all()
        is_offtrack = (poly_dis > track_width / 2.).all()

        # stuck detector data
        self._velocity_queue.append(speed_longitudinal)
        if len(self._velocity_queue) >= self._velocity_stack_size:
            self._velocity_queue.popleft()

        state = {"x": ego_x,
                 "y": ego_y,
                 "heading": yaw,
                 "way_orientation_angle": way_orientation_angle,
                 "delta_yaw": delta_yaw,
                 "distance_from_center": dis,
                 "progress": float(way_i) / way_count,  # from 0 to 1
                 "speed": speed,
                 "speed_longitudinal": speed_longitudinal,
                 "speed_longitudinal_waypoint": speed_lon_waypoint,
                 "steering_angle": steering,
                 "track_width": track_width,
                 "track_length": self._track_length,
                 "current_wp_index": way_i,  # not from deepracer, but is often useful
                 "closest_waypoints": [  # (ahead and back)
                     next_way_i,
                     prev_way_i
                 ],
                 "is_left_of_center": is_left_of_center,
                 "is_reversed": is_reversed,
                 "all_wheels_on_track": all_wheels_on_track,
                 "is_offtrack": is_offtrack,
                 "is_vehicle_front": False,  # TODO implement vehicle hazard
                 "waypoints": self.waypoints,  # TODO think whether we copy it or just linking to same object
                 "is_stuck": all(v < 0.01 for v in self._velocity_queue)
                 }
        return state

    def clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        global global_sensor_list
        for s in global_sensor_list:
            s.stop()
            s.destroy()

        actor_ids = []
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                actor_ids.append(actor.id)
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker' or (
                            actor.type_id.startswith('sensor') and actor.is_listening):
                        actor.stop()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in actor_ids])
        global_sensor_list.clear()

    @staticmethod
    def acc_to_throttle_break(acc):
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 8, 0, 1)
        return throttle, brake

    def add_low_traction_areas(self):
        """Adds square low traction areas to the simulation.
        Parameters of the areas are taken from the config,
        position is random and constant throughout the episode.

        Returns: None
        """
        size = self.exp_config['augmentations']['low_friction_areas_size']
        extent = carla.Location(size, size, 100.0)

        for i in range(self.exp_config['augmentations']['low_friction_n']):
            friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')

            friction_bp.set_attribute(
                'friction', str(self.exp_config['augmentations']['low_friction_multiplier'])
            )
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            transform = carla.Transform()
            location = self._get_random_place_on_track()
            # center the area on the location
            location.x -= extent.x / 2
            location.y -= extent.y / 2
            location.z -= extent.z / 2
            transform.location = location
            self.world.spawn_actor(friction_bp, transform)

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
          filt: the filter indicating what type of actors we'll look at.

        Returns:
          actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = dict()
        for actor in self.world.get_actors().filter(filt):
            poly = misc.get_bounding_polygon(actor)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=(4,)):
        """Try to spawn a surrounding vehicle at specific transform with random blueprint.

        Args:
          transform: the carla transform object.

        Returns:
          Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint(self.other_vehicle_filter, number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        return False

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=(4,)):
        """Create the blueprint for a specific actor type.

        Args:
          actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
          bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
        Args:
          transform: the carla transform object.
        Returns:
          Bool indicating whether the spawn is successful.
        """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self._vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.ego = vehicle
            return True

        return False

    def _get_random_place_on_track(self) -> carla.Location:
        """Calculates a random location on track.

        Returns: A Location of a point on track at 0 z
        """
        wp_idx = random.randint(0, len(self.waypoints) - 1)
        next_idx = (wp_idx + 1) % len(self.waypoints)
        # choose a point within a rectangle bound by inner and outer positions of
        # waypoints at wp_idx and next_idx
        x, y = self._inbound_way[wp_idx]
        perpendicular_vector = self._outbound_way[wp_idx] - self._inbound_way[wp_idx]
        forward_vector = self._inbound_way[next_idx] - self._inbound_way[wp_idx]
        x, y = [x, y] + perpendicular_vector * random.random()
        x, y = [x, y] + forward_vector * random.random()
        return carla.Location(x, y, 0)
