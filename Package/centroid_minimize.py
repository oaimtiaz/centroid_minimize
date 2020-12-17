import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from random import random
from scipy.stats import gaussian_kde


def get_kde(p, x_steps=50, y_steps=50, bandwidth=1000):
    if not isinstance(x_steps, complex):
        x_steps *= 1j
    if not isinstance(y_steps, complex):
        y_steps *= 1j

    x = p[:, 0]
    y = p[:, 1]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    xx, yy = np.mgrid[xmin:xmax:x_steps, ymin:ymax:y_steps]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(xy_train)

    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def put_in_buckets(p, xx, yy, zz):
    x_buckets = np.unique(xx)
    y_buckets = np.unique(yy)
    x_step = x_buckets[1] - x_buckets[0]
    y_step = y_buckets[1] - y_buckets[0]

    zz_scaled = np.copy(zz)
    scaler = MinMaxScaler(feature_range=(-.8, .2))
    zz_scaled = scaler.fit_transform(zz_scaled)
    # zz_scaled = np.tanh(zz_scaled)

    z_vals = np.array([zz_scaled[(x_buckets + x_step >= x) & (x_buckets - x_step <= x)][0][
                           (y_buckets + y_step >= y) & (y_buckets - y_step <= y)][0] for x, y in p])
    return z_vals


def get_weights(p, x_steps=50, y_steps=50, bandwidth=10):
    xx, yy, zz = get_kde(p, x_steps, y_steps, bandwidth)
    weights = 1 - put_in_buckets(p, xx, yy, zz)
    return weights


def calculate_region_center(x_min, x_max, y_min, y_max):
    return x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2


def calculate_point_centroid(p):
    return np.mean(p, axis=0)

def dist(p1, p2):
    return np.linalg.norm(p2 - p1)


def take_samples(p, weights, width, height, x_min=0, y_min=0, n_samples=100, geo_center=None):
    if geo_center is None:
        geo_center = calculate_region_center(x_min, x_min + width, y_min, y_min + height)
    starting_center = calculate_point_centroid(p)
    smallest_dist = dist(starting_center, geo_center)
    samples = None
    for i in range(n_samples):
        sampled_vals = []
        for coord, threshold in zip(p, weights):
            if random() < threshold:
                sampled_vals.append(coord)
        sampled_vals = np.array(sampled_vals)
        sampled_center = calculate_point_centroid(sampled_vals)
        temp_dist = dist(sampled_center, geo_center)
        if temp_dist < smallest_dist:
            smallest_dist = temp_dist
            samples = sampled_vals
        if samples is not None and temp_dist == smallest_dist:
            if sampled_vals.shape[0] > samples.shape[0]:
                samples = sampled_vals
    return samples


def centroid_minimize(p, width=None, height=None, x_min=0, y_min=0, target_dist=None, max_iterations=100,
                      samples_per_iteration=100, min_samples=None, bandwidth=None, x_steps=50, y_steps=50, **kwargs):

    if width is None:
        x_min = np.amin(p[:, 0])
        width = np.amax(p[:, 0]) - x_min
    if height is None:
        y_min = np.amin(p[:, 1])
        height = np.amax(p[:, 1]) - y_min
    if min_samples is None:
        min_samples = p.shape[0] // 40
    if bandwidth is None:
        bandwidth = (width*height) / 20

    geo_center = calculate_region_center(x_min, x_min + width, y_min, y_min + height)
    point_center = calculate_point_centroid(p)

    if target_dist is None:
        target_dist = dist(point_center, geo_center) / 50

    chosen_sampling = p

    for i in range(max_iterations):
        weights = get_weights(chosen_sampling, bandwidth=bandwidth, x_steps=x_steps, y_steps=y_steps)
        best_sample = take_samples(chosen_sampling, weights, width, height, x_min=x_min, y_min=y_min,
                                   geo_center=geo_center, n_samples=samples_per_iteration)
        stagnant_count = 0

        if best_sample is not None:
            stagnant_count = 0
            chosen_sampling = best_sample
            new_center = calculate_point_centroid(best_sample)
            new_dist = dist(new_center, geo_center)

            if chosen_sampling.shape[0] <= min_samples or new_dist < target_dist:
                break
        else:
            stagnant_count += 1
            if stagnant_count > 4:
                break

    return chosen_sampling