import math
import random

import carla
import numpy as np
from absl import flags, app
from carla_birdeye_view import (
    BirdViewProducer,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    BirdViewCropType,
)
from carla_birdeye_view.mask import PixelDimensions
from cv2 import cv2 as cv

from tools.bew import virtual_lidar as vl
from tools.common import CarlaSyncMode

flags.DEFINE_integer('port', 2000, "listen on port")
flags.DEFINE_string('host', default='localhost', help="Simulator host url")
flags.DEFINE_enum('crop_type', 'front_rear', ['front_rear', 'front'], 'crop type for bird eye view')
flags.DEFINE_boolean('is_sync', False, 'Execute ub synchronous mode')
flags.DEFINE_integer('fps', 10, 'FPS in sync mode ')

FLAGS = flags.FLAGS

STUCK_SPEED_THRESHOLD_IN_KMH = 3
MAX_STUCK_FRAMES = 30


def get_speed(actor: carla.Actor) -> float:
    """in km/h"""
    vector: carla.Vector3D = actor.get_velocity()
    MPS_TO_KMH = 3.6
    return MPS_TO_KMH * math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


def main(*args, **kwargs):
    crop_type = BirdViewCropType.FRONT_AND_REAR_AREA if FLAGS.crop_type == 'front_rear' else BirdViewCropType.FRONT_AREA_ONLY
    client = carla.Client(FLAGS.host, FLAGS.port)
    client.set_timeout(6.0)
    world = client.get_world()
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    blueprints = world.get_blueprint_library()

    settings = world.get_settings()
    if FLAGS.is_sync:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / 10.
    else:
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    # hero_bp = random.choice(blueprints.filter("vehicle.audi.a2"))
    hero_bp = random.choice(blueprints.filter("vehicle.kart.kart"))
    hero_bp.set_attribute("role_name", "hero")
    transform = random.choice(spawn_points)
    agent = world.spawn_actor(hero_bp, transform)
    agent.set_autopilot(True)

    birdview_producer = BirdViewProducer(
        client,
        PixelDimensions(width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT),
        pixels_per_meter=4,
        crop_type=crop_type
    )

    vlidar = vl.VirtualLiDAR()

    def do_sim(sync_mode=None):
        stuck_frames_count = 0
        while True:
            # NOTE imshow requires BGR color model
            if sync_mode:
                sync_mode.tick(timeout=1.5)
            bw_masks: np.ndarray = birdview_producer.produce(agent_vehicle=agent)
            distances = vlidar(bw_masks)

            # render
            # bw_bgr = cv.cvtColor(BirdViewProducer.as_rgb(bw_masks), cv.COLOR_BGR2RGB)
            bw_rgb = BirdViewProducer.as_rgb(bw_masks)
            bw_rgb_dist = vlidar.vlidar_rgb(bw_rgb, bw_masks, distances)
            bw_bgr_dist = cv.cvtColor(bw_rgb_dist, cv.COLOR_RGB2BGR)
            cv.imshow("BirdView RGB", bw_bgr_dist)

            # Teleport when stuck for too long
            if get_speed(agent) < STUCK_SPEED_THRESHOLD_IN_KMH:
                stuck_frames_count += 1
            else:
                stuck_frames_count = 0

            if stuck_frames_count == MAX_STUCK_FRAMES:
                agent.set_autopilot(False)
                agent.set_transform(random.choice(spawn_points))
                agent.set_autopilot(True)

            # Play next frames without having to wait for the key
            key = cv.waitKey(10) & 0xFF
            if key == 27:  # ESC
                break

    if FLAGS.is_sync:
        with CarlaSyncMode(world, fps=FLAGS.fps) as sync_mode:
            do_sim(sync_mode)
    else:
        do_sim()
    cv.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
