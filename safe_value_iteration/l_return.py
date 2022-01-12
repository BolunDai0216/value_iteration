from pdb import set_trace

import numpy as np
import torch


class LReturn:
    def __init__(self, **kwargs):
        eps = kwargs.get('eps', 1e-4)
        rho = -np.log(eps) / kwargs.get('T', 5.0)

        self.lam = kwargs.get('lam', 0.85)
        self.gam = np.exp(-rho * kwargs.get('dt', 1.0/125.0))
        self.N = np.ceil(np.log(eps / (1. - self.lam)) /
                         np.log(self.lam)).astype(int)
        self.n_timesteps = int(kwargs.get('T', 5.0) /
                               kwargs.get('dt', 1.0/125.0))

        """
        A self.N = 3 example would be:
                     ｜1  1  1  ｜
        return_mat = ｜0  g  g  ｜
                     ｜0  0  g^2｜
        l_return_vec = [(1-λ), (1-λ)λ, λ^2]
        """
        return_mat = torch.zeros(self.N, self.N)
        return_mat_value = torch.zeros(self.N, 1)
        l_return_vec = torch.zeros(self.N, 1)
        for i in range(self.N):
            return_mat[i, i:] = self.gam**i
            return_mat_value[i] = self.gam**(i+1)
            l_return_vec[i] = self.lam**i * (1 - self.lam)
        l_return_vec[-1] = self.lam**(self.N-1)

        if kwargs.get('cuda', False):
            self.return_mat = return_mat.cuda()
            self.return_mat_value = return_mat_value.cuda()
            self.l_return_vec = l_return_vec.cuda()
        else:
            self.return_mat = return_mat
            self.return_mat_value = return_mat_value
            self.l_return_vec = l_return_vec

    def get_returns(self, rewards, values):
        l_returns = []
        for i in range(self.n_timesteps):
            horizon = np.min([self.N, self.n_timesteps-i])
            i_returns = rewards[:, 0, i:i+horizon] @ \
                self.return_mat[:horizon, :horizon]
            i_returns += values[:, 0, i:i+horizon] @ \
                self.return_mat_value[:horizon, :]
            i_l_returns = i_returns @ self.l_return_vec[:horizon, :]
            l_returns.append(i_l_returns)
        l_returns = torch.cat(l_returns, dim=1).view(-1, self.n_timesteps, 1)

        return l_returns


class ReplayBuffer:
    def __init__(self, **kwargs):
        self.state_buf = []
        self.reward_buf = []
        self.value_buf = []

        self.l_return = LReturn(**kwargs)

    def add(self, state=None, reward=None, value=None):
        """
        No need to save: 
        (1) the value function of the initial state;
        (2) the terminal state.
        """
        if state is not None:
            self.state_buf.append(state)
        if reward is not None:
            self.reward_buf.append(reward)
        if value is not None:
            self.value_buf.append(value)

    def calculate_l_return(self):
        """
                 The state should contain: x0, x1, ..., xN-1
               The rewards should contain: r1, r2, ..., rN
        The value function should contain: V1, V2, ..., VN
        ----------------------------------------------------
        where N = T / dt
        """
        rewards = torch.cat(self.reward_buf, dim=2)
        values = torch.cat(self.value_buf, dim=2)

        # get returns
        l_returns = self.l_return.get_returns(rewards, values)

        return l_returns

    def sample(self):
        states = torch.cat(self.state_buf, dim=2)
        states_dim = states.shape
        states = states.view(states_dim[0], states_dim[-1], -1)
        l_returns = self.calculate_l_return()

        n_states = states.shape[-1]
        states = states.view(-1, n_states)
        l_returns = l_returns.view(-1, 1)

        # reset all buffers
        self.state_buf = []
        self.reward_buf = []
        self.value_buf = []

        return states, l_returns


def main():
    N = 4

    return_mat = torch.zeros(N, N)
    for i in range(N):
        return_mat[i, i:] = 2.0**i

    return_mat_target = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                      [0.0, 2.0, 2.0, 2.0],
                                      [0.0, 0.0, 4.0, 4.0],
                                      [0.0, 0.0, 0.0, 8.0]])

    return_mat_err = torch.sum(return_mat_target - return_mat).item()
    assert return_mat_err <= 1e-9

    lam = 1.0
    N = 4

    l_return_vec = torch.zeros(N, 1)
    for i in range(N):
        l_return_vec[i] = lam**i * (1 - lam)
    l_return_vec[-1] = lam**(N-1)

    l_return_vec_target = torch.tensor([[0.0], [0.0], [0.0], [1.0]])
    l_return_vec_err = torch.sum(l_return_vec - l_return_vec_target).item()
    assert l_return_vec_err <= 1e-9

    lam = 0.0
    N = 5

    l_return_vec = torch.zeros(N, 1)
    for i in range(N):
        l_return_vec[i] = lam**i * (1 - lam)
    l_return_vec[-1] = lam**(N-1)

    l_return_vec_target = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0]])
    l_return_vec_err = torch.sum(l_return_vec - l_return_vec_target).item()
    assert l_return_vec_err <= 1e-9

    lam = 0.5
    N = 5

    l_return_vec = torch.zeros(N, 1)
    for i in range(N):
        l_return_vec[i] = lam**i * (1 - lam)
    l_return_vec[-1] = lam**(N-1)

    l_return_vec_target = torch.tensor(
        [[0.5], [0.25], [0.125], [0.0625], [0.0625]])
    l_return_vec_err = torch.sum(l_return_vec - l_return_vec_target).item()
    assert l_return_vec_err <= 1e-9


if __name__ == "__main__":
    main()
