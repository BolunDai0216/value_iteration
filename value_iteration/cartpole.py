import numpy as np
import torch

from deep_differential_network.utils import jacobian
from value_iteration.cost_functions import ArcTangent, SineQuadraticCost
from value_iteration.pendulum import BaseSystem
CUDA_AVAILABLE = torch.cuda.is_available()


class Cartpole(BaseSystem):
    name = "Cartpole"
    labels = ("x", "theta", "x_dot", "theta_dot")

    def __init__(self, cuda=False, **kwargs):
        super(Cartpole, self).__init__()
        device = torch.device('cuda') if cuda else torch.device('cpu')

        # Define Duration:
        self.T = kwargs.get("T", 10.0)
        self.dt = kwargs.get("dt", 1./500.)

        # Define the System:
        self.n_state = 4
        self.n_act = 1
        self.n_joint = 1
        # number parameters that define the dynamics in the case of cartpole it is three: cart mass, pole mass and pole length
        self.n_parameter = 3

        # Continuous Joints:
        # Right now only one continuous joint is supported
        self.wrap, self.wrap_i = True, 1

        # State Constraints:
        self.x_target = torch.tensor([0.0, 0.0, 0.0, 0.0])
        self.x_start = torch.tensor([0.0, np.pi, 0.01, 0.01])
        self.x_start_var = torch.tensor([1.e-3, 1.e-3, 1.e-6, 1.e-6])
        self.x_lim = torch.tensor([8., np.pi, 8., 8.])
        self.x_init = torch.tensor([0.0, np.pi, 0.01, 0.01])
        self.u_lim = torch.tensor([200., ])

        # Define Dynamics:
        self.gravity = -9.81

        # theta = [cart mass, pole mass, pole length]
        self.theta_min = torch.tensor([0.5, 0.05, 0.25]).to(
            device).view(1, self.n_parameter, 1)
        self.theta = torch.tensor([1., 0.1, 0.5]).to(
            device).view(1, self.n_parameter, 1)
        self.theta_max = torch.tensor([2., 0.15, 0.75]).to(
            device).view(1, self.n_parameter, 1)

        # Test dynamics:
        self.check_dynamics()

        self.device = None
        Pendulum.cuda(self) if cuda else Pendulum.cpu(self)

    def dyn(self, x, dtheta=None, gradient=False):
        cat = torch.cat

        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x).to(
            self.theta.device) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        n_samples = x.shape[0]

        # Update the dynamics parameters with disturbance:
        if dtheta is not None:
            dtheta = torch.from_numpy(dtheta).float() if isinstance(
                dtheta, np.ndarray) else dtheta
            dtheta = dtheta.view(-1, self.n_parameter, 1)
            assert dtheta.shape[0] in (1, n_samples)

            theta = self.theta + dtheta
            theta = torch.min(torch.max(theta, self.theta_min), self.theta_max)

        else:
            theta = self.theta
            theta = theta

    def grad_dyn_theta(self, x):
        pass

    def cuda(self, device=None):
        pass

    def cpu(self):
        pass


def main():
    from deep_differential_network.utils import jacobian

    # GPU vs. CPU:
    cuda = True

    # Seed the test:
    seed = 20
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create system:
    sys = Cartpole()


if __name__ == "__main__":
    main()
