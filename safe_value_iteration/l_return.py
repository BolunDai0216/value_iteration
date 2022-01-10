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

    def get_return(self):
        return 0

    def get_l_return(self):
        return 0


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
        set_trace()
        return 0

    def sample(self):
        return 0


def main():
    pass


if __name__ == "__main__":
    main()
