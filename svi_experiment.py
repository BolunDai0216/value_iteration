import json

import munch
import tensorflow as tf

from safe_value_iteration.envs.pendulum_env import PendulumEnv


class SVI_Train:
    def __init__(self, env):
        self.env = env

    def train(self):
        pass

    def evaluate(self):
        pass

    def rollout(self):
        pass

    def update(self):
        pass

    def get_action(self, safe=False):
        pass


def main():
    env = PendulumEnv(cuda=True)
    alg = SVI_Train(env)


if __name__ == "__main__":
    main()
