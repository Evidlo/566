#!/bin/env python3
# Evan Widloski - 2019-12-14

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def expectation(samples, v, amp_l, pos_l, width):
    """ return P(Z_n = m | {a_m}, {\tau_m}) for each n

    Probability that a given sample belongs to each class

    Args:
        samples (ndarray): (N × M) array of samples
        v (int): drift velocity
        amp_l (ndarray): (M) array of mixture component amplitudes
        pos_l (ndarray): (M) array of mixture component positions
        width: (int): width of mixture component
    """

    shifted_samples = samples[:, 0] - v * samples[:, 1]
    # compute which samples overlap which mixture components
    overlaps = np.abs(
        shifted_samples[:, np.newaxis] - pos_l[np.newaxis, :]
    ) <= (width // 2 + 1)

    # indices of samples not overlapping any rectangle
    unowned = np.product(np.invert(overlaps), axis=1) == 1
    owned = np.invert(unowned)

    # for samples which overlap one or more mixture components
    #   weight expected classes by mixture component amplitude
    expectations = np.zeros(overlaps.shape)
    # expectations[owned] = (
    #     overlaps[owned] / (np.sum(overlaps[owned], axis=1)[:, np.newaxis])
    # )
    expectations[owned] = (
        overlaps[owned] * amp_l /
        (np.sum(overlaps[owned] * amp_l, axis=1)[:, np.newaxis])
    )
    # assign equal probability to each mixture component for samples
    #   that do not overlap any mixture component
    expectations[unowned] = np.ones(amp_l.shape) / len(amp_l)

    return expectations, overlaps


def maximization(samples, expectations):

    v = []
    pos_l = []
    for class_probabilities in expectations.T:

        lr = LinearRegression()
        # LinearRegression needs X to be 2D for some reason
        lr.fit(samples[:, 1, np.newaxis], samples[:, 0], sample_weight=class_probabilities)
        v.append(lr.coef_)
        pos_l.append(lr.intercept_)
    #     v_est, pos_est = np.polyfit(samples[:, 1], samples[:, 0], 1, w=class_probabilities)
    #     v.append(v_est)
    #     pos_l.append(pos_est)

    v = np.mean(v)

    return v, np.array(pos_l, dtype=int)


def compute_lam(pos_l, amp_l, width, sample_grid):

    lam = np.zeros(sample_grid)
    lam[pos_l] = amp_l
    lam = np.convolve(np.ones(width), lam, mode='same')
    lam /= np.sum(amp_l * width)

    return lam


def draw_components(samples, duration, width):
    from matplotlib.patches import Polygon

    fig = plt.figure()
    ax = fig.gca()
    fig.hold(True)
    ax.scatter(samples[:, 0], samples[:, 1])

    plt.show()

    p_l = []

    while True:
        plt.draw()
        v, pos_l = yield

        for p in reversed(ax.patches):
            p.remove()

        for pos in pos_l:
            mid_bot = np.array((pos, 0))
            mid_top = np.array((pos + v * duration, duration))

            xy = np.array((
                mid_bot - (width // 2, 0),
                mid_top - (width // 2, 0),
                mid_top + (width // 2, 0),
                mid_bot + (width // 2, 0),
            ))

            p = Polygon(xy, closed=True, color='red', alpha=0.25 / len(pos_l))
            p_l.append(p)

            ax.add_patch(p)



def draw_lam(lam_true, lam_samples):

    fig = plt.figure()

    ax_true = fig.add_axes((0.1, 0.1, 0.85, 0.85))
    ax_est = fig.add_axes((0.1, 0.1, 0.85, 0.85))

    ax_true.hist(lam_samples, density=True, bins=100, color='orange')
    ax_true.plot(lam_true, color='C0')
    plt_est, = ax_est.plot(lam_true, color='red')

    plt.show()

    while True:

        plt.draw()
        lam_est = yield
        # ax_est.clear()
        plt_est.set_ydata(lam_est)

        plt.draw()


def generate_samples(lam, v, duration, num_samples):

    cum = np.cumsum(lam)

    # generate samples
    lam_samples = np.searchsorted(cum, np.random.uniform(min(cum), max(cum), num_samples))
    times = np.random.uniform(0, duration, num_samples)
    samples = np.vstack((lam_samples + times * v, times)).T

    return samples, lam_samples
