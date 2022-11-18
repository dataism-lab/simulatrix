import os
import shutil
import time

import cv2
import numpy as np
from ray.tune import Stopper


class TimeStopper(Stopper):
    def __init__(self, duration_str: str):
        seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
        duration_seconds = int(duration_str[:-1]) * seconds_per_unit[duration_str[-1]]
        self._start = time.time()
        self._deadline = duration_seconds

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time.time() - self._start > self._deadline


def post_process_image(image, normalized=True, grayscale=True):
    """
    Convert image to gray scale and normalize between -1 and 1 if required
    :param image:
    :param normalized:
    :param grayscale
    :return: image
    """
    if isinstance(image, list):
        image = image[0]
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[:, :, np.newaxis]

    if normalized:
        return (image.astype(np.float32) - 128) / 128
    else:
        return image.astype(np.uint8)


def find_latest_checkpoint(directory):
    """
    Finds the latest checkpoint, based on how RLLib creates and names them.
    """
    start = directory
    max_checkpoint_int = -1
    checkpoint_path = ""

    # 1st layer: Check for the different run folders
    for f in os.listdir(start):
        if os.path.isdir(start + "/" + f):
            temp = start + "/" + f

            # 2nd layer: Check all the checkpoint folders
            for c in os.listdir(temp):
                if "checkpoint_" in c:

                    # 3rd layer: Get the most recent checkpoint
                    checkpoint_int = int(''.join([n for n in c
                                                  if n.isdigit()]))
                    if checkpoint_int > max_checkpoint_int:
                        max_checkpoint_int = checkpoint_int
                        checkpoint_path = temp + "/" + c + "/" + c[:c.index('_')] + '-' + str(max_checkpoint_int)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Could not find any checkpoint, make sure that you have selected the correct folder path."
            "'checkpoint_path': " + checkpoint_path
        )

    return checkpoint_path


def get_checkpoint(name, directory, restore=False, overwrite=False):
    training_directory = os.path.join(directory, name)

    if overwrite and restore:
        raise RuntimeError(
            "Both 'overwrite' and 'restore' cannot be True at the same time")

    if overwrite:
        if os.path.isdir(training_directory):
            shutil.rmtree(training_directory)
            print("Removing all contents inside '" + training_directory + "'")
        return None

    if restore:
        return find_latest_checkpoint(training_directory)

    if os.path.isdir(training_directory) and len(os.listdir(training_directory)) != 0:
        raise RuntimeError(
            "The directory where you are trying to train (" +
            training_directory + ") is not empty. "
                                 "To start a new training instance, make sure this folder is either empty, non-existing "
                                 "or use the '--overwrite' argument to remove all the contents inside"
        )

    return None
