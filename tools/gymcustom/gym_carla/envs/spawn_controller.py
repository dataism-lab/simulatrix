import random
from typing import List, Dict
import carla
import numpy as np

from tools.gymcustom.gym_carla.envs import misc


class SpawnController:
    def __init__(
            self,
            exp_config: Dict,
            waypoints: np.ndarray,
            _inbound_way: np.ndarray,
            seed=42,
    ):
        user_spawn_points = exp_config.get("spawn_points")
        if len(user_spawn_points) > 0:
            self.spawn_transforms = self._from_user_defined(user_spawn_points)
        else:
            spawn_off_center = exp_config['spawn_off_center']
            n_spawn_points = exp_config['n_spawn_points']
            self.spawn_transforms = self._prepare_spawn_points(
                waypoints, _inbound_way, n_spawn_points, seed, spawn_off_center)

        self.spawn_mode = exp_config['spawn_mode']
        if self.spawn_mode == 'consequent':
            self.sp_point_index = 0
            self.consequent_mode_step = exp_config['consequent_mode_step']

    def get_spawn_transform(self) -> carla.Transform:
        if self.spawn_mode == 'random':
            return random.choice(self.spawn_transforms)
        if self.spawn_mode == 'consequent':
            transform = self.spawn_transforms[self.sp_point_index]
            self.sp_point_index = (self.sp_point_index + self.consequent_mode_step) % len(self.spawn_transforms)
            return transform
        raise NotImplementedError(f"Not implemented behavior for {self.spawn_mode}.")

    @staticmethod
    def _from_user_defined(user_spawn_points: List[str]) -> List[carla.Transform]:
        spawn_points = []
        for transform in user_spawn_points:
            x, y, yaw = [float(x) for x in transform.split(",")]
            transform = carla.Transform(
                carla.Location(x, y, 0.5),
                carla.Rotation(0., yaw, 0.)
            )
            spawn_points.append(transform)
        return spawn_points

    @staticmethod
    def _prepare_spawn_points(
            waypoints,
            _inbound_way,
            num_spawns_points=30,
            seed=42,
            spawn_off_center=False
    ) -> List[carla.Transform]:
        """
        Produces a list of transforms that put the car at possible spawn points.

        Args:
          num_spawns_points: number of waypoints to be taken as spawn points
          seed: random seed for waypoint selection
          spawn_off_center: if True, moves the spawning points away from center line.
            Keeps yaw the same.

        Returns:
            The list of carla transforms.
        """
        spawn_points = []

        random.seed(seed)
        # indices = random.sample(range(len(waypoints)), num_spawns_points)
        indices = list(range(len(waypoints)))[::len(waypoints) // num_spawns_points]
        for i in indices:
            x, y = waypoints[i]
            next_wp = waypoints[(i + 1) % len(waypoints)]
            yaw = misc.get_way_direction((x, y), next_wp)

            if spawn_off_center:
                perpendicular_vector = waypoints[i] - _inbound_way[i]
                # -0.5 so that displacement is in both directions
                displacement_coef = random.random() - 0.5
                x += perpendicular_vector[0] * displacement_coef
                y += perpendicular_vector[1] * displacement_coef

            transform = carla.Transform(
                carla.Location(x, y, 0.5),
                carla.Rotation(0., yaw, 0.)
            )
            spawn_points.append(transform)
        return spawn_points
