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
        Cartpole.cuda(self) if cuda else Cartpole.cpu(self)

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

            #          ∂B3     sinθ[M+2m-m(sinθ)^2]
            # dB3dt = ------ = --------------------
            #           ∂θ      l(M + m(sinθ)^2)^2
            dB3dt_numerator = s * (M + 2*m - m * (s**2))
            dB3dt_denominator = da3dt_denominator1

            dB3dt = dB3dt_numerator / dB3dt_denominator

            dadx = cat(
                [
                    cat((zeros, zeros, zeros, zeros), dim=1).unsqueeze(-1),
                    cat((zeros, zeros, da2dt, da3dt), dim=1).unsqueeze(-1),
                    cat((ones, zeros, zeros, zeros), dim=1).unsqueeze(-1),
                    cat((zeros, ones, da2ddt, da3ddt), dim=1).unsqueeze(-1)
                ], dim=1
            ).view(-1, self.n_state, self.n_state)

            dBdx = cat(
                [
                    cat((zeros, zeros, zeros, zeros), dim=1).unsqueeze(-1),
                    cat((zeros, zeros, dB2dt, dB3dt), dim=1).unsqueeze(-1),
                    cat((zeros, zeros, zeros, zeros), dim=1).unsqueeze(-1),
                    cat((zeros, zeros, zeros, zeros), dim=1).unsqueeze(-1)
                ], dim=1
            ).view(-1, self.n_state, self.n_state, self.n_act)

            assert dadx.shape == (n_samples, self.n_state, self.n_state)
            assert dBdx.shape == (n_samples, self.n_state,
                                  self.n_state, self.n_act)
            out = (a, B, dadx, dBdx)

        if is_numpy:
            out = [array.cpu().detach().numpy() for array in out]

        return out

    def grad_dyn_theta(self, x):
        is_numpy = True if isinstance(x, np.ndarray) else False
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x.view(-1, self.n_state, 1)
        n_samples = x.shape[0]

        c = torch.cos(x[:, 1])
        s = torch.sin(x[:, 1])
        M = self.theta[:, 0]
        m = self.theta[:, 1]
        l = self.theta[:, 2]

        dadth = torch.zeros(n_samples, self.n_parameter,
                            self.n_state).to(x.device)

        #           ∂a2    -ml(2dθsinθ+2gcosθsinθ)
        # da2dM =  ----- = -----------------------
        #           ∂M       2[-ml(cosθ)^2+M+m]^2
        da2dM_numerator = -m*l*(2*x[:, 3]*s + 2*self.gravity*c*s)
        da2dM_denominator = 2 * (-m*l*(c**2)+M+m)**2
        da2dM = da2dM_numerator / da2dM_denominator

        #           ∂a2        Mlsinθ(dθ+gcosθ)
        # da2dm =  ----- = -----------------------
        #           ∂m       [-ml(cosθ)^2+M+m]^2
        da2dm_numerator = M*l*s*(x[:, 3] + self.gravity*c)
        da2dm_denominator = da2dM_denominator / 2
        da2dm = da2dm_numerator / da2dm_denominator

        #           ∂a2     msinθ(M+m)(dθ+g*cosθ)
        # da2dl =  ----- = -----------------------
        #           ∂l       [-ml(cosθ)^2+M+m]^2
        da2dl_numerator = m*s*(M+m)*(x[:, 3] + self.gravity*c)
        da2dl_denominator = da2dm_denominator
        da2dl = da2dl_numerator / da2dl_denominator

        #           ∂a3    mcosθsinθ[l(dθ)^2+gcosθ]
        # da3dM =  ----- = -----------------------
        #           ∂M        l[m(sinθ)^2+M]^2
        da3dM_numerator = m*c*s*(l*(x[:, 3]**2) + self.gravity*c)
        da3dM_denominator = l*(m*(s**2)+M)**2
        da3dM = da3dM_numerator / da3dM_denominator

        #           ∂a3    -Mcosθsinθ[l(dθ)^2+gcosθ]
        # da3dm =  ----- = -------------------------
        #           ∂m         l[m(sinθ)^2+M]^2
        da3dm_numerator = -M*c*s*(l*(x[:, 3]**2) + self.gravity*c)
        da3dm_denominator = l*(m*(s**2)+M)**2
        da3dm = da3dm_numerator / da3dm_denominator

        #           ∂a3      (M+m)gsinθ
        # da3dl =  ----- = ----------------
        #           ∂l     l^2[m(sinθ)^2+M]
        da3dl_numerator = (M+m)*self.gravity*s
        da3dl_denominator = (l**2)*(m*(s**2)+M)
        da3dl = da3dl_numerator / da3dl_denominator

        dadth[:, 0, 2] = da2dM.squeeze()
        dadth[:, 1, 2] = da2dm.squeeze()
        dadth[:, 2, 2] = da2dl.squeeze()
        dadth[:, 0, 3] = da3dM.squeeze()
        dadth[:, 1, 3] = da3dm.squeeze()
        dadth[:, 2, 3] = da3dl.squeeze()

        dBdth = torch.zeros(n_samples, self.n_parameter,
                            self.n_state, self.n_act).to(x.device)

        #           ∂B2            -1
        # dB2dM =  ----- = -------------------
        #           ∂M     [-ml(cosθ)^2+M+m]^2
        dB2dM_denominator = (M+m-m*l*(c**2))**2
        dB2dM = -1 / dB2dM_denominator

        #           ∂B2       l(cosθ)^2-1
        # dB2dm =  ----- = -------------------
        #           ∂m     [-ml(cosθ)^2+M+m]^2
        dB2dm_numerator = l * (c**2) - 1
        dB2dm_denominator = dB2dM_denominator
        dB2dm = dB2dm_numerator / dB2dm_denominator

        #           ∂B2         m(cosθ)^2
        # dB2dl =  ----- = -------------------
        #           ∂l     [-ml(cosθ)^2+M+m]^2
        dB2dl_numerator = m * (c**2)
        dB2dl_denominator = dB2dM_denominator
        dB2dl = dB2dl_numerator / dB2dl_denominator

        #           ∂B3         cosθ
        # dB3dM =  ----- = ----------------
        #           ∂M     l[m(sinθ)^2+M]^2
        dB3dM_denominator = l*(M+m*(s**2))**2
        dB3dM = c / dB3dM_denominator

        #           ∂B3      cosθ(sinθ)^2
        # dB3dm =  ----- = ----------------
        #           ∂m     l[m(sinθ)^2+M]^2
        dB3dm_numerator = c * (s**2)
        dB3dm_denominator = dB3dM_denominator
        dB3dm = dB3dm_numerator / dB3dm_denominator

        #           ∂B3          cosθ
        # dB3dl =  ----- = ----------------
        #           ∂l     l^2[m(sinθ)^2+M]
        dB3dl_denominator = (l**2)*(M+m*(s**2))
        dB3dl = c / dB3dl_denominator

        dBdth[:, 0, 2] = dB2dM
        dBdth[:, 1, 2] = dB2dm
        dBdth[:, 2, 2] = dB2dl
        dBdth[:, 0, 3] = dB3dM
        dBdth[:, 1, 3] = dB3dm
        dBdth[:, 2, 3] = dB3dl

        out = dadth, dBdth

        if is_numpy:
            out = [array.numpy() for array in out]

        return out

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


class CartpoleLogCos(Cartpole):
    name = "Cartpole_LogCosCost"

    def __init__(self, Q, R, cuda=False, **kwargs):

        # Create the dynamics:
        super(CartpoleLogCos, self).__init__(cuda=cuda, **kwargs)
        self.u_lim = torch.tensor([10.0, ])

        assert Q.size == self.n_state and np.all(Q > 0.0)
        self.Q = np.diag(Q).reshape((self.n_state, self.n_state))

        assert R.size == self.n_act and np.all(R > 0.0)
        self.R = np.diag(R).reshape((self.n_act, self.n_act))

        # Create the Reward Function:
        self.q = SineQuadraticCost(
            self.Q, np.array([1.0, 1.0, 0.0, 0.0]), cuda=cuda)

        # Determine beta s.t. the curvature at u = 0 is identical to 2R
        beta = (4. * self.u_lim[0] ** 2 / np.pi * self.R)[0, 0].item()
        self.r = ArcTangent(alpha=self.u_lim.numpy()[0], beta=beta)

    def rwd(self, x, u):
        return self.q(x) + self.r(u)

    def cuda(self, device=None):
        super(CartpoleLogCos, self).cuda(device=device)
        self.q.cuda(device=device)
        return self

    def cpu(self):
        super(CartpoleLogCos, self).cpu()
        self.q.cpu()
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

    n_samples = 10
    x_lim = torch.from_numpy(sys.x_lim).float() if isinstance(
        sys.x_lim, np.ndarray) else sys.x_lim
    x_test = torch.distributions.uniform.Uniform(
        -x_lim, x_lim).sample((n_samples,))
    # x_test = torch.tensor([np.pi / 2., 0.5]).view(1, sys.n_state, 1)

    dtheta = torch.zeros(1, sys.n_parameter, 1)

    if cuda:
        sys, x_test, dtheta = sys.cuda(), x_test.cuda(), dtheta.cuda()

    ###################################################################################################################
    # Test dynamics gradient w.r.t. state:
    dadx_shape = (n_samples, sys.n_state, sys.n_state)
    dBdx_shape = (n_samples, sys.n_state, sys.n_state, sys.n_act)

    a, B, dadx, dBdx = sys.dyn(x_test, gradient=True)

    dadx_auto = torch.cat([jacobian(lambda x: sys.dyn(
        x)[0], x_test[i:i+1]) for i in range(n_samples)], dim=0)
    dBdx_auto = torch.cat([jacobian(lambda x: sys.dyn(
        x)[1], x_test[i:i+1]) for i in range(n_samples)], dim=0)

    err_a = (dadx_auto.view(dadx_shape) - dadx).abs().sum() / n_samples
    err_B = (dBdx_auto.view(dBdx_shape) - dBdx).abs().sum() / n_samples
    assert err_a <= 1.e-5 and err_B <= 1.e-6

    ###################################################################################################################
    # Test dynamics gradient w.r.t. model parameter:
    dadp_shape = (n_samples, sys.n_parameter, sys.n_state)
    dBdp_shape = (n_samples, sys.n_parameter, sys.n_state, sys.n_act)

    dadp, dBdp = sys.grad_dyn_theta(x_test)

    dadp_auto = torch.cat([jacobian(lambda x: sys.dyn(x_test[i], dtheta=x)[
                          0], dtheta) for i in range(n_samples)], dim=0)
    dBdp_auto = torch.cat([jacobian(lambda x: sys.dyn(x_test[i], dtheta=x)[
                          1], dtheta) for i in range(n_samples)], dim=0)

    err_a = (dadp_auto.view(dadp_shape) - dadp).abs().sum() / n_samples
    err_B = (dBdp_auto.view(dBdp_shape) - dBdp).abs().sum() / n_samples

    assert err_a <= 1.e-5 and err_B <= 1.e-6


if __name__ == "__main__":
    main()
