# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : nms.py
# @Time    : 2019/3/7 上午11:19
# @Desc    :

import numpy as np

import config as cfg


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)  # include: False, not include: True


def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor


def region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))   # ( , 0, [1,2,3,...])
    return D


def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))
    return rows


def nms(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})
    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 1]
                if score >= threshold:
                    ith_score = predict[ij[0], ij[1], 2:3]
                    if not (cfg.trunc_threshold <= ith_score < 1 -
                            cfg.trunc_threshold):
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * cfg.pixel_size
                        py = (ij[0] + 0.5) * cfg.pixel_size
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7], (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
    return score_list, quad_list


def nms_hor(predict, activation_pixels, threshold=0.9, trunc_threshold=0.1, pixel_size=4, epsilon=1e-4):
    h, w, _ = predict.shape
    region_list = []
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})
    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        min_x, min_y, max_x, max_y = (np.inf, np.inf, 0, 0)
        for row in group:
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 1]
                px = (ij[1] + 0.5) * pixel_size
                py = (ij[0] + 0.5) * pixel_size

                if px < min_x: min_x = px
                if px > max_x: max_x = px
                if py < min_y: min_y = py
                if py > max_y: max_y = py

                if score >= threshold:
                    ith_score = predict[ij[0], ij[1], 2:3]
                    if not (trunc_threshold <= ith_score < 1 - trunc_threshold):
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7], (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v

        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + epsilon)

        # TODO: change to support any direction
        # one side vertex(only horizontal)
        if score_list[g_th, 0] == 0 and score_list[g_th, 3] != 0:
            score_list[g_th, 0:2] += 1
            quad_list[g_th, 0, 0] = min_x - (max_y - min_y) / 3 if min_x - (max_y - min_y) / 3 > 0 else 0
            quad_list[g_th, 1, 0] = quad_list[g_th, 0, 0]
            quad_list[g_th, 0, 1] = quad_list[g_th, 3, 1]
            quad_list[g_th, 1, 1] = quad_list[g_th, 2, 1]

        elif score_list[g_th, 3] == 0 and score_list[g_th, 0] != 0:
            score_list[g_th, 2:4] += 1
            quad_list[g_th, 2, 0] = max_x + (max_y - min_y) / 3 if min_x + (max_y - min_y) / 3 < w * 4 else w * 4
            quad_list[g_th, 3, 0] = quad_list[g_th, 2, 0]
            quad_list[g_th, 2, 1] = quad_list[g_th, 1, 1]
            quad_list[g_th, 3, 1] = quad_list[g_th, 0, 1]

        if total_score[0, 0] != 0 and total_score[3, 0] != 0:
            if quad_list[g_th, 0, 0] - min_x > 0 and quad_list[g_th, 3, 0] - max_x < 0:
                quad_list[g_th, 0, 0] = min_x - (max_y - min_y) / 3 if min_x - (max_y - min_y) / 3 > 0 else 0
                quad_list[g_th, 1, 0] = quad_list[g_th, 0, 0]
                quad_list[g_th, 0, 1] = min_y - (max_y - min_y) / 3 if min_y - (max_y - min_y) / 3 > 0 else 0
                quad_list[g_th, 1, 1] = max_y + (max_y - min_y) / 3 if min_y - (max_y - min_y) / 3 < h * 4 else h * 4

                quad_list[g_th, 2, 0] = max_x + (max_y - min_y) / 3 if min_x + (max_y - min_y) / 3 < w * 4 else w * 4
                quad_list[g_th, 3, 0] = quad_list[g_th, 2, 0]
                quad_list[g_th, 2, 1] = quad_list[g_th, 1, 1]
                quad_list[g_th, 3, 1] = quad_list[g_th, 0, 1]

            if quad_list[g_th, 0, 0] - min_x > 0:
                quad_list[g_th, 0, 0] = min_x - (max_y - min_y) / 3 if min_x - (max_y - min_y) / 3 > 0 else 0
                quad_list[g_th, 1, 0] = quad_list[g_th, 0, 0]
                quad_list[g_th, 0, 1] = quad_list[g_th, 3, 1]
                quad_list[g_th, 1, 1] = quad_list[g_th, 2, 1]

            if quad_list[g_th, 3, 0] - max_x < 0:
                quad_list[g_th, 2, 0] = max_x + (max_y - min_y) / 3 if min_x + (max_y - min_y) / 3 < w * 4 else w * 4
                quad_list[g_th, 3, 0] = quad_list[g_th, 2, 0]
                quad_list[g_th, 2, 1] = quad_list[g_th, 1, 1]
                quad_list[g_th, 3, 1] = quad_list[g_th, 0, 1]

    return score_list, quad_list