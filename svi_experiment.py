import json
from pdb import set_trace

import munch
import numpy as np
import tensorflow as tf
import torch

from safe_value_iteration.envs.pendulum_env import PendulumEnv
from safe_value_iteration.value_function_model import ValueFunctionModel
from safe_value_iteration.l_return import ReplayBuffer


class SVI_Train:
    def __init__(self, env, **kwargs):
        self.env = env

        self.model = ValueFunctionModel(
            self.env.n_state,
            feature=self.env.feature_mask,
            **kwargs,
        )

        self.buf = ReplayBuffer(**kwargs)

        self.hyper = kwargs
        self.safe = kwargs.get("safe", False)

    def train(self, **kwargs):
        pass

    def evaluate(self, **kwargs):
        reward, terminal_state = self.rollout(**kwargs)

        return reward, terminal_state

    def rollout(self, **kwargs):
        state = self.env.reset(
            type=kwargs.get("init_type", "uniform"),
            is_numpy=kwargs.get("is_numpy", False),
        )

        n_iters = int(self.env.T / self.env.dt)
        mean_reward = 0.0

        for iter in range(n_iters):
            V, dVdx, action = self.get_action(
                state, safe=self.safe
            )
            next_state, reward = self.env.step(action)

            if iter == 0:
                self.buf.add(state=state, reward=reward)
            else:
                self.buf.add(state=state, reward=reward, value=V)

            mean_reward += torch.mean(reward).item()
            state = next_state

        # Get value function of terminal state
        V, _ = self.model(state)
        self.buf.add(value=V)

        mean_terminal_state = torch.mean(state, dim=0)

        return mean_reward, mean_terminal_state

    def update(self, **kwargs):
        pass

    def get_action(self, state, safe=False):
        # Get V and dVdx
        V, dVdx = self.model(state)
        dVdx = dVdx.transpose(dim0=1, dim1=2)
        # Get G
        G = self.env.G(state)
        GT = G.transpose(dim0=1, dim1=2)
        # Get Action
        GT_dVdx = torch.matmul(GT, dVdx)
        action = self.env.r.grad_convex_conjugate(GT_dVdx)

        if safe:
            print("----- Not yet implemented -----")

        return V, dVdx, action


def main():
    hyper = {
        # Network
        'n_network': 4,
        'activation': 'Tanh',
        'n_width': 96,
        'n_depth': 3,
        'n_output': 1,
        'g_hidden': 1.41,
        'g_output': 1.,
        'b_output': -0.1,

        # Experiment
        'init_type': 'uniform',
        'n_batch': 2,
        'is_numpy': False,
        'cuda': True,
        'safe': False,
        'T': 5.0,
        'dt': 1.0 / 125.0,
        'lam': 0.85,
        'eps': 1e-4,
    }
    env = PendulumEnv(**hyper)
    alg = SVI_Train(env, **hyper)

    alg.rollout(**hyper)
    alg.buf.calculate_l_return()


if __name__ == "__main__":
    main()
