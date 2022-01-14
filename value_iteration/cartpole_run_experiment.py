import pickle
import time
from pdb import set_trace

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from deep_differential_network.replay_memory import (PyTorchReplayMemory,
                                                     PyTorchTestMemory)

from value_iteration.sample_rollouts import sample_data
from value_iteration.update_value_function import (eval_memory,
                                                   update_value_function)
from value_iteration.utils import add_nan, linspace
from value_iteration.value_function import ValueFunctionMixture


def _sample_rollout(value_fun, hyper, system, run_config):
    t0 = time.perf_counter()

    # Perform the roll-out
    n_samples = int(hyper["test_minibatch"])
    n_steps = int(np.ceil(hyper['T'] * run_config['fs_return']))
    n_trajectories = int(np.ceil(n_samples / n_steps))
    mem_data, trajectory_data = sample_data(
        hyper['T'], n_trajectories, value_fun, hyper, system, run_config)
    R = trajectory_data[3].squeeze()
    R_mean = torch.mean(R).item()
    R_std = torch.std(R).item()

    # Compute distribution around last time-step
    x_n_avg = torch.mean(torch.abs(trajectory_data[0][-1, :, :, 0]), dim=0)
    x_n_std = torch.std(torch.abs(trajectory_data[0][-1, :, :, 0]), dim=0)
    x_n = zip(x_n_avg, x_n_std)

    t_comp = time.perf_counter() - t0
    return (mem_data, trajectory_data, (R_mean, R_std), t_comp, x_n)


def run_experiment(hyper):
    cuda = torch.cuda.is_available()
    alg_name = "rFVI" if hyper['robust'] else "cFVI"

    # Configuration for sampling trajectories from the system
    run_config = {"verbose": False, 'mode': 'init', 'fs_return': 10.,
                  'x_noise': hyper['x_noise'], 'u_noise': hyper['u_noise']}

    # Build the dynamical system:
    Q = np.array([float(x) for x in hyper['state_cost'].split(',')])
    R = np.array([float(x) for x in hyper['action_cost'].split(',')])
    system = hyper['system_class'](Q, R, cuda=cuda, **hyper)

    # Compute Gamma s.t., the weight of the reward at time T is \eps, i.e., exp(-rho T) = gamma^(T/\Delta t) = eps:
    rho = -np.log(hyper['eps']) / hyper["T"]
    hyper["gamma"] = np.exp(-rho * hyper["dt"])

    # Construct Value Function:
    feature = torch.zeros(system.n_state)
    if system.wrap:
        feature[system.wrap_i] = 1.0

    val_fun_kwargs = {'feature': feature}
    value_fun = ValueFunctionMixture(system.n_state, **val_fun_kwargs, **hyper)

    if hyper['checkpoint'] is not None:
        data = torch.load(hyper['checkpoint'],
                          map_location=torch.device('cpu'))

        hyper = data['hyper']
        hyper['n_iter'] = 0
        hyper['plot'] = True
        hyper['save_plot_data'] = True

        value_fun = ValueFunctionMixture(
            system.n_state, **val_fun_kwargs, **data['hyper'])
        value_fun.load_state_dict(data["state_dict"])

    value_fun = value_fun.cuda() if cuda else value_fun.cpu()

    print("\n\n################################################")
    print(f"{'Sample Data:':>25}", end="\t")
    t0_data = time.perf_counter()

    # Sample uniformly from the n-d hypercube
    n_samples = hyper["eval_minibatch"]
    x_lim = torch.from_numpy(system.x_lim).float() if isinstance(
        system.x_lim, np.ndarray) else system.x_lim

    x = torch.distributions.uniform.Uniform(-x_lim,
                                            x_lim).sample((n_samples, ))
    x = x.view(-1, system.n_state,
               1).float().cuda() if cuda else x.view(-1, system.n_state, 1).float()

    ax, Bx, dadx, dBdx = system.dyn(x, gradient=True)
    mem_data = [x, ax, dadx, Bx, dBdx]

    # Memory Dimensions:
    mem_dim = ((system.n_state, 1),                                 # x
               (system.n_state, 1),                                 # a(x)
               (system.n_state, system.n_state),                    # da(x)/dx
               (system.n_state, system.n_act),                      # B(x)
               (system.n_state, system.n_state, system.n_act))      # dB(x)dx

    # Generate Replay Memory:
    mem = PyTorchReplayMemory(mem_data[0].shape[0], min(
        mem_data[0].shape[0], int(hyper["eval_minibatch"]/2)), mem_dim, cuda)

    mem.add_samples(mem_data)

    print(f"{time.perf_counter() - t0_data:05.2e}s")
    print("\n\n################################################")
    print("Learn the Value Function:")

    t0_training = time.perf_counter()
    step_i = -1

    try:
        for step_i in range(hyper["n_iter"]):
            t0_iter = time.perf_counter()

            # Compute the roll-out:
            args = (value_fun, hyper, system, run_config)
            mem_data, uniform_trajectory_data, R_uniform, t_rollout, x_last = _sample_rollout(
                *args)
            t_wait = 0.0

            # Update the Value Function:
            out = update_value_function(
                step_i, value_fun.cuda(), system, mem, hyper, None)
            value_fun, _, _ = out

            print("Rollout Computation:")
            str_x_n = "[" + \
                ", ".join(
                    [f"{x[0]:.2f} \u00B1 {x[1]:.2f}" for x in x_last]) + "]"
            print(f"x_0 reward = {R_uniform[0]:+.2f} \u00B1 {1.96*R_uniform[1]:.2f}, x_N = {str_x_n} "
                  f"Comp Time = {t_rollout:.2f}s, Wait Time = {t_wait:.2f}s\n"
                  f"Iteration Time = {time.perf_counter() - t0_iter:.2f}s")
            print("")

            # Sample new data:
            if hyper['mode'] == 'RTDP':
                mem.add_samples(mem_data)

            # Save the model:
            if np.mod(step_i+1, 25) == 0:
                model_file = f"data/{alg_name}_step_{step_i+1:03d}.torch"
                torch.save({"epoch": step_i, "hyper": hyper,
                           "state_dict": value_fun.state_dict()}, model_file)

    except KeyboardInterrupt as err:
        t_train = time.perf_counter() - t0_training
        print(
            f"Training stopped due to Keyboard Interrupt. Comp Time = {t_train:.2f}s\n")

    finally:
        # Training Time:
        t_train = time.perf_counter() - t0_training

        # Save the Model:
        if step_i > 10:
            model_file = f"data/{alg_name}.torch"
            torch.save({"epoch": step_i, "hyper": hyper,
                       "state_dict": value_fun.state_dict()}, model_file)

    print("\n################################################")
    print("Evaluate the Value Function:")
    t0 = time.perf_counter()
    n_test = 100

    # Evaluate expected reward with downward initial state distribution:
    test_config = {"verbose": False, 'mode': 'test',
                   'fs_return': 100., 'x_noise': 0.0, 'u_noise': 0.0}
    _, downward_trajectory_data = sample_data(
        hyper["T"], n_test, value_fun, hyper, system, test_config)
    R_downward = downward_trajectory_data[3].squeeze()
    R_downward_mean = torch.mean(R_downward).item()
    R_downward_std = torch.std(R_downward).item()

    print("\nPerformance:")
    print(
        f"Expected Reward - Downward = {R_downward_mean:.2f} \u00B1 {1.96*R_downward_std:.2f}")

    print("\n################################################")
    print("\n################################################")
    print("Plot the Value Function:")

    if hyper["plot"]:
        n_plot = 20
        scale = 1.0

        x_tra = downward_trajectory_data[0].cpu().numpy()
        u_tra = downward_trajectory_data[1].cpu().numpy()
        t_tra = np.arange(x_tra.shape[0]) * hyper["dt"]  # time

        fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        lw = 0.2

        for i in range(x_tra.shape[1]):
            axs[0].plot(t_tra, x_tra[:, i, 0, 0],
                        color="cornflowerblue", linewidth=lw)
            axs[1].plot(t_tra, x_tra[:, i, 1, 0],
                        color="cornflowerblue", linewidth=lw)
            axs[2].plot(t_tra, x_tra[:, i, 2, 0],
                        color="cornflowerblue", linewidth=lw)
            axs[3].plot(t_tra, x_tra[:, i, 3, 0],
                        color="cornflowerblue", linewidth=lw)
            axs[4].plot(t_tra, u_tra[:, i, 0, 0],
                        color="cornflowerblue", linewidth=lw)

        ylabels = ["Cart Position [m]", "Pole Angle [rad]",
                   "Cart Velocity [m/s]", "Pole Angular Velocity [rad/s]", "Control [N]"]
        for i in range(5):
            axs[i].axhline(0.0, color="darkorange",
                           linestyle="dashed", zorder=-20)
            axs[i].set_ylabel(ylabels[i], labelpad=0.1, fontsize=12)
            axs[i].yaxis.set_label_coords(-0.08, 0.5)

        axs[1].axhline(np.pi, color="black", linestyle="dashed", zorder=-20)
        axs[1].axhline(-np.pi, color="black", linestyle="dashed", zorder=-20)
        axs[-1].set_xlabel("Time [s]", fontsize=12)

        plt.savefig(f"figures/trajectories_{system.name}.png", dpi=200,
                    transparent=False, bbox_inches="tight")
