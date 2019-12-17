#!/bin/env python3
# Evan Widloski - 2019-12-14

import numpy as np
import matplotlib.pyplot as plt

from em import compute_lam, draw_lam, draw_components, expectation, maximization, generate_samples

# %% generate

sample_grid = 1000

# num rectangles
M = 10
# drift velocity
v = 200
# num samples
num_samples = 1000000
# simulation duration
duration = 1

# rectangle amplitudes and initial positions
width = sample_grid // 4
np.random.seed(2)
# amp_l = np.random.random(M)
amp_l = np.ones(M)
pos_l = np.random.randint(width // 2, sample_grid - width // 2, M)
pos_l = np.linspace(200, 800, M, dtype=int)
# pos_l = np.array((250, 375))
# amp_l = np.array((1, 1))

# 1D intensity function
lam = compute_lam(pos_l, amp_l, width, sample_grid)

samples, lam_samples = generate_samples(lam, v, duration, num_samples)


# plt.hold(True)
# plt.plot(lam)
# plt.hist(lam_samples, density=True, bins=100)

# plt.figure()
# plt.scatter(samples[:1000, 0], samples[:1000, 1])

# plt.show()

# %% em

v_est = 0
# pos_est = np.random.randint(0, 1000, M)
# pos_est = np.array((lam_samples.min(), lam_samples.max()))
pos_est = np.linspace(lam_samples.min(), lam_samples.max(), M)

drawing_comp = draw_components(samples[:1000], duration, width)
next(drawing_comp)
drawing_lam = draw_lam(lam, lam_samples)
next(drawing_lam)
# fig_scat = plt.figure()
# ax_scat = fig_scat.gca()

for _ in range(100):
    e, _ = expectation(samples, v_est, amp_l, pos_est, width)
    v_est, pos_est = maximization(samples, e)
    e, _ = expectation(samples, v_est, amp_l, pos_est, width)

    drawing_comp.send((v_est, pos_est))
    print(v_est, pos_est)
    lam_est = compute_lam(pos_est, amp_l, width, sample_grid)
    drawing_lam.send(lam_est)

    # comp1 = samples[:1000][e[:1000, 0] == 1]
    # comp2 = samples[:1000][e[:1000, 1] == 1]
    # comp12 = samples[:1000][e[:1000, 0] == 0.5]
    # colors = np.empty(1000, dtype='object')
    # colors[e[:1000, 0] == 1] = 'blue'
    # colors[e[:1000, 1] == 1] = 'red'
    # colors[e[:1000, 0] == 0.5] = 'purple'
    # ax_scat.scatter(samples[:1000, 0], samples[:1000, 1], c=colors)

    plt.show()
    plt.pause(.05)

    # import ipdb
    # ipdb.set_trace()

