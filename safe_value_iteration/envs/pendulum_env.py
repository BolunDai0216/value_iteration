from pdb import set_trace

import numpy as np
import torch
from safe_value_iteration.envs.base_env import BaseEnv
from value_iteration.cost_functions import ArcTangent, SineQuadraticCost


class PendulumEnv(BaseEnv):
    def __init__(self, cuda=False, **kwargs):
        super(PendulumEnv, self).__init__()

        # Define Duration
        self.T = kwargs.get("T", 5.0)
        self.dt = kwargs.get("dt", 1./125.)
        self.n_step = int(self.T / self.dt)

        # Define System Parameters
        self.n_batch = kwargs.get("n_batch", 64)
        self.n_state = 2
        self.n_act = 1
        self.wrap = True  # if there exists an element of the state that is a joint angle
        self.wrap_i = 0   # which element of the state is a joint angle
        self.m = 1.0      # pendulum mass
        self.l = 1.0      # pendulum length
        self.feature_mask = torch.zeros(self.n_state)
        self.feature_mask[self.wrap_i] = 1.0

        # Target & Initial State
        self.x_target = torch.tensor([0.0, 0.0])
        self.x_init = torch.tensor([np.pi, 0.0])
        self.x_init_var = torch.tensor([1.e-3, 1.e-6])
        # sampling range of initial state
        self.x_init_lim = torch.tensor([np.pi, 8.])

        # State Limits (not enforced in the env)
        self.x_lim = torch.tensor([np.pi, 8.])

        # Control Limits
        self.u_lim = torch.tensor([200., ])

        # Rewards
        self.Q = np.diag([1.0, 0.1]).reshape((self.n_state, self.n_state))
        self.R = np.diag([0.5]).reshape((self.n_act, self.n_act))
        self.q = SineQuadraticCost(self.Q, np.array([1.0, 0.0]), cuda=cuda)
        beta = (4. * self.u_lim[0] ** 2 / np.pi * self.R)[0, 0].item()
        self.r = ArcTangent(alpha=self.u_lim.numpy()[0], beta=beta)

        PendulumEnv.cuda(self) if cuda else PendulumEnv.cpu(self)

    def step(self, action):
        is_numpy = True if isinstance(action, np.ndarray) else False
        assert is_numpy == self.is_numpy

        F = self.F(self.state)
        G = self.G(self.state)

        # next_state = self.state + \
        #     (F + G @ (action + self.n_u[self.step_n])
        #      ) * self.dt + self.n_x[self.step_n]
        next_state = self.state + (F + G @ action) * self.dt

        if self.wrap:
            next_state[:, self.wrap_i, :] = torch.remainder(
                next_state[:, self.wrap_i, :] + np.pi, 2 * np.pi) - np.pi

        reward = self.reward(next_state, action)

        self.state = next_state
        self.step_n += 1

        return next_state, reward

    def reset(self, type="uniform", is_numpy=False):
        if type == "uniform":
            # Sample uniformly from a given range of the state space
            dist_x = torch.distributions.uniform.Uniform(
                -self.x_init_lim, self.x_init_lim)
        elif type == "downward":
            # Sample using a normal distribution centering the pendulum pointing downwards
            sigma = torch.diag(self.x_start_var.float()).view(
                self.n_state, self.n_state)
            dist_x = torch.distributions.multivariate_normal.MultivariateNormal(
                self.x_init, sigma)

        # Sample and reshape
        x0 = dist_x.sample((self.n_batch,))
        x0 = x0.view(-1, self.n_state, 1).to(self.device)

        if self.wrap:
            x0[:, self.wrap_i, :] = torch.remainder(
                x0[:, self.wrap_i, :] + np.pi, 2 * np.pi) - np.pi

        if is_numpy:
            # Transform to numpy array
            x0 = x0.cpu().detach().numpy()

        self.state = x0

        # boolean value for determining the type of current episode
        # if True then all outputs are np.ndarray
        # if False then all outputs are torch.Tensor
        self.is_numpy = is_numpy

        # Noise
        self.get_noise()

        # Step counter
        self.step_n = 0

        return x0

    def F(self, state):
        is_numpy = True if isinstance(state, np.ndarray) else False
        assert is_numpy == self.is_numpy

        dtheta = state[:, 1, :]

        if is_numpy:
            ddtheta = 1.5 * (9.81 / self.l) * np.sin(state[:, 0, :])
            F = np.concatenate([dtheta, ddtheta],
                               axis=1).reshape(-1, self.n_state, 1)
        else:
            ddtheta = 1.5 * (9.81 / self.l) * torch.sin(state[:, 0, :])
            F = torch.cat([dtheta, ddtheta], dim=1)
            F = F.view(-1, self.n_state, 1).to(self.device)

        return F

    def G(self, state):
        is_numpy = True if isinstance(state, np.ndarray) else False
        assert is_numpy == self.is_numpy

        if is_numpy:
            zeros = np.zeros_like(state[:, 0, :])
            ddtheta = np.ones_like(state[:, 0, :]) * 3 / (self.m * self.l)
            G = np.concatenate(
                [zeros, ddtheta], axis=1).reshape(-1, self.n_state, self.n_act)
        else:
            zeros = torch.zeros_like(state[:, 0, :])
            ddtheta = torch.ones_like(state[:, 0, :]) * 3 / (self.m * self.l)
            G = torch.cat([zeros, ddtheta], dim=1)
            G = G.view(-1, self.n_state, self.n_act).to(self.device)

        return G

    def reward(self, state, action):
        is_numpy = True if isinstance(state, np.ndarray) else False
        assert is_numpy == self.is_numpy

        reward = -(self.q(state) + self.r(action)) * self.dt

        return reward

    def get_noise(self, x_noise=0.0, u_noise=0.0):
        # State noise: Wiener Process
        mu_x, eye_x = torch.zeros(self.n_state), torch.eye(self.n_state)
        xi_x_alpha = x_noise * self.x_lim.view(1, 1, self.n_state, 1)
        dist_x_noise = torch.distributions.multivariate_normal.MultivariateNormal(
            mu_x, covariance_matrix=eye_x)
        n_x = dist_x_noise.sample((self.n_step, self.n_batch)).float().to(
            self.device).view(self.n_step, self.n_batch, self.n_state, 1)
        self.n_x = xi_x_alpha.to(n_x.device) / 1.96 * np.sqrt(self.dt) * n_x

        # Action noise: Ornstein–Uhlenbeck process
        mu_u, eye_u = torch.zeros(self.n_act), torch.eye(self.n_act)
        theta = 0.5
        t = torch.arange(0.0, self.T, self.dt)
        exp_minus = torch.exp(-theta * t).view(-1, 1, 1, 1).to(self.n_x.device)
        exp_plus = torch.exp(theta * t).view(-1, 1, 1, 1).to(self.n_x.device)

        xi_u_alpha = u_noise * self.u_lim
        dist_u_noise = torch.distributions.multivariate_normal.MultivariateNormal(
            mu_u, covariance_matrix=eye_u)
        n_u = dist_u_noise.sample((self.n_step, self.n_batch)).float().to(
            self.device).view(self.n_step, self.n_batch, self.n_act, 1)
        self.n_u = xi_u_alpha.to(n_u.device) / 1.96 * exp_minus * \
            torch.cumsum(exp_plus * np.sqrt(self.dt) * n_u, dim=0)

    def cuda(self, device=None):
        self.u_lim = self.u_lim.cuda(device=device)
        self.device = torch.device('cuda')
        return self

    def cpu(self):
        self.u_lim = self.u_lim.cpu()
        self.device = torch.device('cpu')
        return self


def main():
    env = PendulumEnv(cuda=True)

    # Test Code using torch.Tensor
    x0_torch = env.reset()
    action_torch = torch.ones(env.n_batch, 1, 1).to(env.device)
    F_torch = env.F(x0_torch)
    G_torch = env.G(x0_torch)

    assert isinstance(x0_torch, torch.Tensor)
    assert isinstance(F_torch, torch.Tensor)
    assert isinstance(G_torch, torch.Tensor)

    assert x0_torch.shape == (env.n_batch, env.n_state, 1)
    assert F_torch.shape == (env.n_batch, env.n_state, 1)
    assert G_torch.shape == (env.n_batch, env.n_state, env.n_act)

    next_state_torch, reward_torch = env.step(action_torch)

    assert isinstance(next_state_torch, torch.Tensor)
    assert isinstance(reward_torch, torch.Tensor)
    assert next_state_torch.shape == (env.n_batch, env.n_state, 1)
    assert reward_torch.shape == (env.n_batch, 1, 1)

    # Test Code using np.ndarray
    x0_np = env.reset(is_numpy=True)
    action_np = np.ones([env.n_batch, 1, 1])
    F_np = env.F(x0_np)
    G_np = env.G(x0_np)

    assert isinstance(x0_np, np.ndarray)
    assert isinstance(F_np, np.ndarray)
    assert isinstance(G_np, np.ndarray)

    assert x0_np.shape == (env.n_batch, env.n_state, 1)
    assert F_np.shape == (env.n_batch, env.n_state, 1)
    assert G_np.shape == (env.n_batch, env.n_state, env.n_act)

    next_state_np, reward_np = env.step(action_np)

    assert isinstance(next_state_np, np.ndarray)
    assert isinstance(reward_np, np.ndarray)
    assert next_state_np.shape == (env.n_batch, env.n_state, 1)
    assert reward_np.shape == (env.n_batch, 1, 1)


if __name__ == "__main__":
    main()
