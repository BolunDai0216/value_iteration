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
        self.n_l_returns = int(kwargs.get('T', 5.0) /
                               kwargs.get('dt', 1.0/125.0))

        """
        A self.N = 3 example would be:
                     ｜1  1  1  ｜
        return_mat = ｜0  g  g  ｜
                     ｜0  0  g^2｜
        l_return_vec = [(1-λ), (1-λ)λ, λ^2]
        """
        return_mat = torch.zeros(self.N, self.N)
        l_return_vec = torch.zeros(self.N, 1)
        for i in range(self.N):
            return_mat[i, i:] = self.gam**i
            l_return_vec[i] = self.lam**i * (1 - self.lam)
        l_return_vec[-1] = self.lam**(self.N-1)

        if kwargs.get('cuda', False):
            self.return_mat = return_mat.cuda()
            self.l_return_vec = l_return_vec.cuda()
        else:
            self.return_mat = return_mat
            self.l_return_vec = l_return_vec

    def get_returns(self, rewards, values):
        l_returns = []
        for i in range(self.n_l_returns):
            horizon = np.min([self.N, self.n_l_returns-i])
            i_returns = rewards[:, 0, i:i+horizon] @ \
                self.return_mat[:horizon, :horizon]
            i_returns += values[:, 0, i:i+horizon]
            i_l_returns = i_returns @ self.l_return_vec[:horizon, :]
            l_returns.append(i_l_returns)
        l_returns = torch.cat(l_returns, dim=1).view(-1, self.n_l_returns, 1)

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
        l_returns = self.calculate_l_return()

        # reset all buffers
        self.state_buf = []
        self.reward_buf = []
        self.value_buf = []

        return states, l_returns


def main():
    pass


if __name__ == "__main__":
    main()
