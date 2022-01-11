import json
from pdb import set_trace

import munch
import numpy as np
import tensorflow as tf
import torch
from torch.optim import Adam

from safe_value_iteration.envs.pendulum_env import PendulumEnv
from safe_value_iteration.l_return import ReplayBuffer
from safe_value_iteration.value_function_model import ValueFunctionModel


class SVI_Train:
    def __init__(self, env, **kwargs):
        self.env = env

        self.model = ValueFunctionModel(
            self.env.n_state, feature=self.env.feature_mask, **kwargs,
        )

        self.buf = ReplayBuffer(**kwargs)

        self.hyper = kwargs
        self.safe = kwargs.get("safe", False)

    def train(self, **kwargs):
        self.optimizer = Adam(self.model.net.parameters(),
                              lr=kwargs.get("lr_SGD", 3e-5),
                              weight_decay=kwargs.get("weight_decay", 1e-6),
                              amsgrad=True)

        for i in range(kwargs.get("n_epochs", 200)):
            mean_reward, mean_terminal_state = self.rollout(**kwargs)
            loss = self.update(**kwargs)

            print(
                f"Iteration {i}, mean_reward {mean_reward}, mean_terminal_state {mean_terminal_state} , loss {loss}")

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
        states, l_returns = self.buf.sample()

        # set_trace()
        # for j in range(5):
        #     for i in range(20):
        #         states_sample = states[i*125:(i+1)*125, :].detach()
        #         l_returns_sample = l_returns[i*125:(i+1)*125, :].detach()

        for i in range(40):
            loss = self.optimize(states.detach(), l_returns.detach())

        return loss.item()

    def optimize(self, states, targets):
        self.optimizer.zero_grad()
        est, _ = self.model(states, fit=True)
        loss = torch.mean(
            torch.abs(est.squeeze(2) - targets.unsqueeze(0)), dim=0)
        loss = torch.mean(loss)
        loss.backward()
        self.optimizer.step()

        return loss

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
    torch.cuda.set_device(1)

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
        'n_batch': 4,
        'is_numpy': False,
        'cuda': True,
        'safe': False,
        'T': 5.0,
        'dt': 1.0 / 125.0,
        'lam': 0.85,
        'eps': 1e-4,

        # Network Optimization
        'max_iterations': 20,
        'lr_SGD': 1e-4,
        'weight_decay': 1.e-6,
        'exp': 1.,
        'n_epochs': 1000,
    }
    env = PendulumEnv(**hyper)
    alg = SVI_Train(env, **hyper)
    alg.train(**hyper)


if __name__ == "__main__":
    main()
