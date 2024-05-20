#!/usr/bin/env python

import numpy as np
import itertools
import functools
import cv2
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import Mplp


def create_factor(var_names, var_vals, params, feats, obs):
    f_vals_shape = [len(vals) for vals in var_vals]
    f_vals = []

    for vals in itertools.product(*var_vals):
        cur_f_val = sum(params[fi] * cur_feat(*vals, obs) for fi, cur_feat in enumerate(feats))
        f_vals.append(np.exp(cur_f_val))

    f_vals = np.array(f_vals).reshape(f_vals_shape)
    return DiscreteFactor(var_names, f_vals_shape, f_vals)


def unary_feat(x, obs):
    obs = obs * 0.9 + 0.05
    return np.log(obs[x])


def pairwise_feat(ch, xi, xj, obs):
    beta = 0.05
    if xi == xj:
        return 0
    else:
        diff = obs[0][ch] - obs[1][ch]
        return -np.exp(-beta * diff * diff)


def read_images(idx):
    image = cv2.imread(f'export/image_{idx:04d}.jpg')
    labels_gt_im = cv2.imread(f'export/labels_{idx:04d}.png', cv2.IMREAD_ANYDEPTH).astype(np.int8)
    labels_gt_im[labels_gt_im == 65535] = -1
    prob_im = cv2.imread(f'export/prob_{idx:04d}.png', cv2.IMREAD_ANYDEPTH) / 65535.0
    segments_im = cv2.imread(f'export/segments_{idx:04d}.png', cv2.IMREAD_ANYDEPTH)
    return image, labels_gt_im, prob_im, segments_im


def process_image(image, labels_gt_im, prob_im, segments_im, map_size, pixels_thresh):
    mean_rgb = np.zeros([map_size, map_size, 3], dtype=float)
    num_pixels = np.zeros([map_size, map_size], dtype=int)
    prob = np.zeros([map_size, map_size, 2], dtype=float)
    labels_gt = -1 * np.ones([map_size, map_size], dtype=int)

    for y in range(map_size):
        for x in range(map_size):
            cur_seg = y * map_size + x
            cur_pixels = np.nonzero(segments_im == cur_seg)

            if len(cur_pixels[0]) > 0:
                num_pixels[y, x] = len(cur_pixels[0])
                mean_rgb[y, x, :] = np.flip(np.mean(image[cur_pixels], axis=0))
                prob[y, x, 0] = np.mean(prob_im[cur_pixels])
                prob[y, x, 1] = 1.0 - prob[y, x, 0]
                labels_unique, count = np.unique(labels_gt_im[cur_pixels], return_counts=True)
                labels_gt[y, x] = labels_unique[np.argmax(count)]

    return mean_rgb, num_pixels, prob, labels_gt


def build_graph(map_size, pixels_thresh, num_pixels, prob, mean_rgb, unary_feat, pairwise_feat):
    nodes = ['x_' + str(y) + '_' + str(x) for y in range(map_size) for x in range(map_size) if num_pixels[y, x] > pixels_thresh]

    var_to_factor_idx = {}
    factors_u = []
    for y in range(map_size):
        for x in range(map_size):
            if num_pixels[y, x] > pixels_thresh:
                var = 'x_' + str(y) + '_' + str(x)
                cur_f = create_factor([var], [[0, 1]], [0.945212], [unary_feat], prob[y, x])
                var_to_factor_idx[var] = len(factors_u)
                factors_u.append(cur_f)

    factors_p = []
    edges_p = []
    for y in range(map_size - 1):
        for x in range(map_size - 1):
            if num_pixels[y, x] > pixels_thresh:
                if num_pixels[y + 1, x] > pixels_thresh:
                    cur_f_r = create_factor(['x_' + str(y) + '_' + str(x), 'x_' + str(y + 1) + '_' + str(x)],
                                            [[0, 1], [0, 1]], [1.86891, 1.07741, 1.89271],
                                            [functools.partial(pairwise_feat, 0),
                                             functools.partial(pairwise_feat, 1),
                                             functools.partial(pairwise_feat, 2)],
                                            [mean_rgb[y, x], mean_rgb[y + 1, x]])
                    factors_p.append(cur_f_r)
                    edges_p.append(('x_' + str(y) + '_' + str(x), 'x_' + str(y + 1) + '_' + str(x)))

                if num_pixels[y, x + 1] > pixels_thresh:
                    cur_f_c = create_factor(['x_' + str(y) + '_' + str(x), 'x_' + str(y) + '_' + str(x + 1)],
                                            [[0, 1], [0, 1]], [1.86891, 1.07741, 1.89271],
                                            [functools.partial(pairwise_feat, 0),
                                             functools.partial(pairwise_feat, 1),
                                             functools.partial(pairwise_feat, 2)],
                                            [mean_rgb[y, x], mean_rgb[y, x + 1]])
                    factors_p.append(cur_f_c)
                    edges_p.append(('x_' + str(y) + '_' + str(x), 'x_' + str(y) + '_' + str(x + 1)))

    G = MarkovNetwork()
    G.add_nodes_from(nodes)
    G.add_factors(*factors_u)
    G.add_factors(*factors_p)
    G.add_edges_from(edges_p)

    return G, var_to_factor_idx


def infer_labels(G, var_to_factor_idx, num_pixels, map_size, pixels_thresh):
    class_infer = Mplp(G)
    q = class_infer.map_query()

    labels_infer = -1 * np.ones([map_size, map_size], dtype=int)
    for y in range(map_size):
        for x in range(map_size):
            if num_pixels[y, x] > pixels_thresh:
                var = 'x_' + str(y) + '_' + str(x)
                labels_infer[y, x] = q[var]

    return labels_infer


def classify_labels(prob, num_pixels, map_size, pixels_thresh):
    labels_class = -1 * np.ones([map_size, map_size], dtype=int)
    for y in range(map_size):
        for x in range(map_size):
            if num_pixels[y, x] > pixels_thresh:
                labels_class[y, x] = 0 if prob[y, x, 0] >= 0.5 else 1
    return labels_class


def transfer_labels_to_image(labels, segments_im, map_size):
    labels_im = -1 * np.ones_like(segments_im)
    for y in range(map_size):
        for x in range(map_size):
            if labels[y, x] >= 0:
                cur_seg = y * map_size + x
                cur_pixels = np.nonzero(segments_im == cur_seg)
                labels_im[cur_pixels] = labels[y, x]
    return labels_im


def visualize_results(image, labels_infer_im, labels_class_im, labels_gt_im, colors):
    def expand_labels(labels, image_shape):
        expanded_labels = np.zeros((*image_shape, 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            expanded_labels[labels == i] = color
        return expanded_labels

    labels_infer_rgb = expand_labels(labels_infer_im, image.shape[:2])
    labels_class_rgb = expand_labels(labels_class_im, image.shape[:2])
    labels_gt_rgb = expand_labels(labels_gt_im, image.shape[:2])

    image_infer_vis = (0.75 * image + 0.25 * labels_infer_rgb).astype(np.uint8)
    image_class_vis = (0.75 * image + 0.25 * labels_class_rgb).astype(np.uint8)
    image_gt_vis = (0.75 * image + 0.25 * labels_gt_rgb).astype(np.uint8)

    cv2.imshow('inferred', image_infer_vis)
    cv2.imshow('classified', image_class_vis)
    cv2.imshow('ground truth', image_gt_vis)


def main():
    map_size = 50
    pixels_thresh = 500
    num_images = 70

    num_incorrect = 0

    for idx in range(num_images):
        print(f'\n\nImage {idx}')

        image, labels_gt_im, prob_im, segments_im = read_images(idx)
        cv2.imshow('original image', image)
        cv2.waitKey(100)

        mean_rgb, num_pixels, prob, labels_gt = process_image(image, labels_gt_im, prob_im, segments_im, map_size, pixels_thresh)
        G, var_to_factor_idx = build_graph(map_size, pixels_thresh, num_pixels, prob, mean_rgb, unary_feat, pairwise_feat)

        print('Check model:', G.check_model())

        labels_infer = infer_labels(G, var_to_factor_idx, num_pixels, map_size, pixels_thresh)
        labels_class = classify_labels(prob, num_pixels, map_size, pixels_thresh)

        cnt_corr = np.sum(np.logical_and(labels_gt == labels_infer, labels_gt != -1))
        print('Accuracy =', cnt_corr / np.sum(labels_gt != -1))
        num_incorrect += np.sum(labels_gt != -1) - cnt_corr

        labels_infer_im = transfer_labels_to_image(labels_infer, segments_im, map_size)
        labels_class_im = transfer_labels_to_image(labels_class, segments_im, map_size)

        colors = np.array([[0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        visualize_results(image, labels_infer_im, labels_class_im, labels_gt_im, colors)
        cv2.waitKey(100)

    print('Incorrectly inferred', num_incorrect, 'segments')


if __name__ == '__main__':
    main()
