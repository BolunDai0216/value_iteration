import numpy as np
import torch


class BaseEnv:
    def __init__(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass
