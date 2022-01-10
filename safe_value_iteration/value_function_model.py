from pdb import set_trace

import numpy as np
import torch
from value_iteration.value_function import TrigonometricQuadraticNetwork


class ValueFunctionModel:
    def __init__(self, n_state, feature=None, **kwargs):
        """
        **kwargs contains:
           feature: a torch.Tensor of 1 and 0 denoting which element of the state is joint angle
         n_network: how many networks in the ensemble
           n_width: size of hidden layer
          n_hidden: number of hidden layers
          n_output: size of output
        activation: type of activation function
            w_init: how weights are initialized
          b_hidden: what value bias is initialized for hidden layer
          b_output: what value bias is initialized for output layer
          g_hidden: what value hidden gain is initialized (no idea what this is)
          g_output: what value output gain is initialized (no idea what this is)
          p_sparse: (no idea what this is)
        """
        self.net = TrigonometricQuadraticNetwork(
            n_state, feature=feature, **kwargs)
        self.name = self.net.name + f"_mixture{self.net.n_network:02d}"
        self.device = self.net.device

    def __call__(self, state):
        return self.forward(state)

    def forward(self, state):
        V_ensemble, dVdx_ensemble = self.net(state)
        V, dVdx = (V_ensemble.mean(dim=0), dVdx_ensemble.mean(dim=0))

        return V, dVdx

    def cuda(self, device=None):
        self.net.cuda()
        self.device = self.net.device
        return self

    def cpu(self):
        self.net.cpu()
        self.device = self.net.device
        return self

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict):

        try:
            self.net.load_state_dict(state_dict)

        except AttributeError:
            assert len(state_dict) == self.net.n_network

            new_state_dict = {}
            for key in state_dict[0].keys():
                stacked_parameters = torch.stack(
                    [dict_i[key] for dict_i in state_dict], dim=0)
                new_state_dict[key] = stacked_parameters

            self.net.load_state_dict(new_state_dict)


def main():
    # Testing for PendulumEnv
    feature = torch.zeros(2)
    feature[0] = 1.0
    model_hyper = {
        # Network
        'n_network': 4,
        'activation': 'Tanh',
        'n_width': 96,
        'n_depth': 3,
        'n_output': 1,
        'g_hidden': 1.41,
        'g_output': 1.,
        'b_output': -0.1,

        # The rest are chosen to be the default
    }

    model = ValueFunctionModel(2, feature=feature, **model_hyper)
    state = torch.ones(64, 2, 1)
    V, dVdx = model(state)

    assert isinstance(V, torch.Tensor)
    assert isinstance(dVdx, torch.Tensor)
    assert V.shape == (64, 1, 1)
    assert dVdx.shape == (64, 1, 2)

    print("----- Model test passed -----")


if __name__ == "__main__":
    main()
