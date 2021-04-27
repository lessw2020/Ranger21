import numpy as np

# from https://arxiv.org/abs/2103.01338v1


def cheb_steps(m, M, T):
    C, R = (M + m) / 2.0, (M - m) / 2.0
    thetas = (np.arange(T) + 0.5) / T * np.pi
    return 1.0 / (C - R * np.cos(thetas))


def cheb_perm(T):
    perm = np.array([0])
    while len(perm) < T:
        perm = np.vstack([perm, 2 * len(perm) - 1 - perm]).T.flatten()
    return perm


# steps = cheb_steps(0.1,1,8)
# perm = cheb_perm(8)
# schedule = steps[perm]