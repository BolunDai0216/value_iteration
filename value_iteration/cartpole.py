from pdb import set_trace

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

        c = torch.cos(x[:, 1])
        s = torch.sin(x[:, 1])
        M = theta[:, 0]
        m = theta[:, 1]
        l = theta[:, 2]

        # mglsinθcosθ + mldθsinθ
        # ----------------------
        #     M+m-ml(cosθ)^2
        dd_pos_a_numerator = m * self.gravity * l * s * c + m * l * x[:, 3] * s
        dd_pos_a_denominator = M + m - m * l * c**2
        dd_pos_a = dd_pos_a_numerator / dd_pos_a_denominator

        # -(M+m)gsinθ - ml(dθ^2)sinθcosθ
        # ------------------------------
        #        l(M + m(sinθ)^2)
        dd_ang_a_numerator = -(M + m) * self.gravity * \
            s - m * l * (x[:, 3]**2) * s * c
        dd_ang_a_denominator = l * (M + m * (s**2))
        dd_ang_a = dd_ang_a_numerator / dd_ang_a_denominator

        #            1
        # ----------------------
        #     M+m-ml(cosθ)^2
        dd_pos_B_denominator = dd_pos_a_denominator
        dd_pos_B = 1.0 / dd_pos_B_denominator

        #              -cosθ
        # ------------------------------
        #        l(M + m(sinθ)^2)
        dd_ang_B_denominator = dd_ang_a_denominator
        dd_ang_B = -c / dd_ang_B_denominator

        a = torch.cat([x[:, 2], x[:, 3], dd_pos_a, dd_ang_a],
                      dim=1).view(-1, self.n_state, 1)
        B = torch.zeros(x.shape[0], self.n_state,
                        self.n_act).to(self.theta.device)
        B[:, 2] = dd_pos_B
        B[:, 3] = dd_ang_B

        assert a.shape == (n_samples, self.n_state, 1)
        assert B.shape == (n_samples, self.n_state, self.n_act)
        out = (a, B)

        if gradient:
            zeros, ones = torch.zeros_like(x[:, 1]), torch.ones_like(x[:, 1])

            #          ∂a2     ml(dθcosθ-g+2g(cosθ)^2)  (ml)^2cosθsinθ(2dθsinθ+2gcosθsinθ)
            # da2dt = ------ = ---------------------- - ---------------------------------
            #           ∂θ         M+m-ml(cosθ)^2              (M+m-ml(cosθ)^2)^2
            da2dt_numerator1 = m * l * \
                (x[:, 3] * c - self.gravity + 2 * self.gravity * c**2)
            da2dt_numerator2 = ((m * l)**2) * c * s * \
                (2 * x[:, 3] * s + 2 * self.gravity * c * s)

            da2dt_denominator1 = dd_pos_a_denominator
            da2dt_denominator2 = dd_pos_a_denominator**2

            da2dt = (da2dt_numerator1 / da2dt_denominator1) - \
                (da2dt_numerator2 / da2dt_denominator2)

            #           ∂a2          mlsinθ
            # da2ddt = ------- = --------------
            #           ∂(dθ)    M+m-ml(cosθ)^2
            da2ddt_denominator = da2dt_denominator1
            da2ddt = (m * l * s) / da2ddt_denominator

            #          ∂a3     (2msinθcosθ)[mlsinθcosθ(dθ)^2+(M+m)gsinθ]   ml(dθ)^2(cosθ)^2 - ml(dθ)^2(sinθ)^2 + (M+m)gcosθ
            # da3dt = ------ = ---------------------------------------- - ------------------------------------------------
            #           ∂θ                l(M + m(sinθ)^2)^2                              l(M + m(sinθ)^2)
            da3dt_numerator1 = (2*m*s*c) * \
                (m*l*s*c*(x[:, 3]**2)+(M+m)*self.gravity*s)
            da3dt_numerator2 = m*l*(x[:, 3]**2) * \
                (c**2 - s**2) + (M+m)*self.gravity*c

            da3dt_denominator1 = l * (M + m * (s**2))**2
            da3dt_denominator2 = dd_ang_a_denominator

            da3dt = (da3dt_numerator1 / da3dt_denominator1) - \
                (da3dt_numerator2 / da3dt_denominator2)

            #            ∂a3      -2ml(dθ)sinθcosθ
            # da3ddt = -------- = ----------------
            #            ∂(dθ)    l(M + m(sinθ)^2)
            da3ddt_numerator = -2 * m * l * x[:, 3] * s * c
            da3ddt_denominator = dd_ang_a_denominator

            da3ddt = da3ddt_numerator / da3ddt_denominator

            #          ∂B2        -2mlsinθcosθ
            # dB2dt = ------ = ------------------
            #           ∂θ     (M+m-ml(cosθ)^2)^2
            dB2dt_numerator = -2 * m * l * s * c
            dB2dt_denominator = da2dt_denominator1 ** 2

            dB2dt = dB2dt_numerator / dB2dt_denominator

            #          ∂B3     -sinθ[M+2m-m(sinθ)^2]
            # dB3dt = ------ = ---------------------
            #           ∂θ       l(M + m(sinθ)^2)^2
            dB3dt_numerator = -s * (M + 2*m - m * (s**2))
            dB3dt_denominator = da3dt_denominator1

            dB3dt = dB3dt_numerator / dB3dt_denominator

            dadx = cat(
                [
                    cat((zeros, zeros, ones, zeros), dim=1).unsqueeze(-1),
                    cat((zeros, zeros, zeros, ones), dim=1).unsqueeze(-1),
                    cat((zeros, da2dt, zeros, da2ddt), dim=1).unsqueeze(-1),
                    cat((zeros, da3dt, zeros, da3ddt), dim=1).unsqueeze(-1)
                ]
            ).view(-1, self.n_state, self.n_state)

            dBdx = cat(
                [
                    cat((zeros, zeros, zeros, zeros), dim=1).unsqueeze(-1),
                    cat((zeros, zeros, zeros, zeros), dim=1).unsqueeze(-1),
                    cat((zeros, dB2dt, zeros, zeros), dim=1).unsqueeze(-1),
                    cat((zeros, dB3dt, zeros, zeros), dim=1).unsqueeze(-1)
                ]
            ).view(-1, self.n_state, self.n_state, self.n_act)

            assert dadx.shape == (n_samples, self.n_state, self.n_state)
            assert dBdx.shape == (n_samples, self.n_state,
                                  self.n_state, self.n_act)
            out = (a, B, dadx, dBdx)

        if is_numpy:
            out = [array.cpu().detach().numpy() for array in out]

        return out

    def grad_dyn_theta(self, x):
        pass

    def cuda(self, device=None):
        self.theta_min = self.theta_min.cuda(device=device)
        self.theta = self.theta.cuda(device=device)
        self.theta_max = self.theta_max.cuda(device=device)

        self.u_lim = self.u_lim.cuda(device=device)
        self.x_lim = self.x_lim.cuda(device=device)
        self.device = self.theta.device
        return self

    def cpu(self):
        self.theta_min = self.theta_min.cpu()
        self.theta = self.theta.cpu()
        self.theta_max = self.theta_max.cpu()

        self.u_lim = self.u_lim.cpu()
        self.x_lim = self.x_lim.cpu()
        self.device = self.theta.device
        return self


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
