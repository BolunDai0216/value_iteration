import numpy as np
import torch
from safe_value_iteration.envs.base_env import BaseEnv
from pdb import set_trace


class PendulumEnv(BaseEnv):
    def __init__(self, cuda=False, **kwargs):
        super(PendulumEnv, self).__init__()

        # Define Duration
        self.T = kwargs.get("T", 5.0)
        self.dt = kwargs.get("dt", 1./125.)

        # Define System Parameters
        self.n_batch = 64
        self.n_state = 2
        self.n_act = 1
        self.wrap = True  # if there exists an element of the state that is a joint angle
        self.wrap_i = 0   # which element of the state is a joint angle
        self.m = 1.0      # pendulum mass
        self.l = 1.0      # pendulum length

        # Target & Initial State
        self.x_target = torch.tensor([0.0, 0.0])
        self.x_init = torch.tensor([np.pi, 0.0])
        self.x_init_var = torch.tensor([1.e-3, 1.e-6])
        # sampling range of initial state
        self.x_init_lim = torch.tensor([np.pi, 8.])

        # Control Limits
        self.u_lim = torch.tensor([200., ])

        PendulumEnv.cuda(self) if cuda else PendulumEnv.cpu(self)

    def step(self, action):
        is_numpy = True if isinstance(action, np.ndarray) else False
        pass

    def reset(self, type="uniform", batch_size=64, is_numpy=False):
        if type == "uniform":
            dist_x = torch.distributions.uniform.Uniform(
                -self.x_init_lim, self.x_init_lim)
        elif type == "downward":
            sigma = torch.diag(self.x_start_var.float()).view(
                self.n_state, self.n_state)
            dist_x = torch.distributions.multivariate_normal.MultivariateNormal(
                self.x_init, sigma)

        x0 = dist_x.sample((self.n_batch,))
        x0 = x0.view(-1, self.n_state, 1).to(self.device)

        if is_numpy:
            # Transform to numpy array
            x0 = x0.cpu().detach().numpy()

        return x0

    def F(self, state):
        is_numpy = True if isinstance(state, np.ndarray) else False
        pass

    def G(self, state):
        is_numpy = True if isinstance(state, np.ndarray) else False
        pass

    def reward(self, state):
        is_numpy = True if isinstance(state, np.ndarray) else False
        pass

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
    x0_torch = env.reset()

    assert isinstance(x0_torch, torch.Tensor)
    assert x0_torch.shape == (env.n_batch, env.n_state, 1)

    x0_np = env.reset(is_numpy=True)

    assert isinstance(x0_np, np.ndarray)
    assert x0_np.shape == (env.n_batch, env.n_state, 1)


if __name__ == "__main__":
    main()
