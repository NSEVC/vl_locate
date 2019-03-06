# -*- coding: utf-8 -*-
# @Author  : chenlijuan
# @File    : data.py
# @Time    : 2019/3/6 下午2:19
# @Desc    :

# coding:utf-8
import glob
import cv2
import time
import os
import numpy as np

from data_util import GeneratorEnqueuer
import config as cfg


def distort_color(im_data):
    # contrast
    im_data = np.uint8(np.clip((1.5 * im_data + 10), 0, 255))

    return im_data


def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        a = xy_list * ([py, px] - yx_list)
        b = a[:, 0] - a[:, 1]
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False

def point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge):
    nth = -1
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth


def get_images():
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(cfg.training_data_path, '*.{}'.format(ext))))
    return files


def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)

    with open(p, 'r') as f:
        for line in f:
            line = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

            label = line[-1]
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)

        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,`
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im_data, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im_data.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im_data, polys, tags

    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)

        # the cropped area too small
        if xmax - xmin < cfg.min_crop_side_ratio*w or ymax - ymin < cfg.min_crop_side_ratio*h:
            continue

        # select the polygons in cropped area
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []

        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im_data[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue

        im_data = im_data[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im_data, polys, tags

    return im_data, polys, tags


def shrink(xy_list, ratio=cfg.shrink_ratio):
    if ratio == 0.0:
        return xy_list, xy_list
    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))
    # determine which are long or short edges
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
    short_edge = 1 - long_edge
    # cal r length array
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
    # cal theta array
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += cfg.epsilon
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
    # shrink two long edges
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)
    # shrink two short edges
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)
    return temp_new_xy_list, new_xy_list, long_edge


def shrink_edge(xy_list, new_xy_list, edge, r, theta, ratio=cfg.shrink_ratio):
    if ratio == 0.0:
        return
    start_point = edge
    end_point = (edge + 1) % 4
    long_start_sign_x = np.sign(
        xy_list[end_point, 0] - xy_list[start_point, 0])
    new_xy_list[start_point, 0] = \
        xy_list[start_point, 0] + \
        long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])
    long_start_sign_y = np.sign(
        xy_list[end_point, 1] - xy_list[start_point, 1])
    new_xy_list[start_point, 1] = \
        xy_list[start_point, 1] + \
        long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])
    # long edge one, end point
    long_end_sign_x = -1 * long_start_sign_x
    new_xy_list[end_point, 0] = \
        xy_list[end_point, 0] + \
        long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])
    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = \
        xy_list[end_point, 1] + \
        long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])

def process_label(im_size, polys, tags):
    h, w = im_size
    gt_map = np.zeros((h // cfg.pixel_size, w // cfg.pixel_size, 7))
    for xy_list in polys:
        _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
        shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)

        p_min = np.amin(shrink_xy_list, axis=0)
        p_max = np.amax(shrink_xy_list, axis=0)

        # floor of the float
        ji_min = (p_min / cfg.pixel_size - 0.5).astype(int) - 1  # 缩小后框的坐标取整
        # +1 for ceil of the float and +1 for include the end
        ji_max = (p_max / cfg.pixel_size - 0.5).astype(int) + 3

        # 边界保护
        imin = np.maximum(0, ji_min[1])
        imax = np.minimum(h // cfg.pixel_size, ji_max[1])
        jmin = np.maximum(0, ji_min[0])
        jmax = np.minimum(w // cfg.pixel_size, ji_max[0])

        for i in range(imin, imax):
            for j in range(jmin, jmax):
                px = (j + 0.5) * cfg.pixel_size
                py = (i + 0.5) * cfg.pixel_size
                if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):  # 点在框内
                    gt_map[i, j, 0] = 1
                    line_width, line_color = 1, 'red'
                    ith = point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge)
                    vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                    if ith in range(2):
                        gt_map[i, j, 1] = 1
                        gt_map[i, j, 2:3] = ith
                        gt_map[i, j, 3:5] = xy_list[vs[long_edge][ith][0]] - [px, py]
                        gt_map[i, j, 5:] = xy_list[vs[long_edge][ith][1]] - [px, py]

    return gt_map



def generator(input_size=512, batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=False):
    image_list = np.array(get_images())
    print('{} training images in {}'.format(image_list.shape[0], cfg.training_data_path))

    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        images = []
        image_files = []
        gt_maps = []
        for i in index:
            try:
                image_file = image_list[i]
                im_data = cv2.imread(image_file)
                # im_data = distort_color(im_data)
                h, w, _ = im_data.shape
                txt_fn = image_file.replace(os.path.basename(image_file).split('.')[1], 'txt')
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue

                text_polys, text_tags = load_annoataion(txt_fn)

                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

                # if text_polys.shape[0] == 0:
                #     continue
                # random scale this image
                rd_scale = np.random.choice(random_scale)
                im_data = cv2.resize(im_data, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale

                # random crop a area from image, contain at least one text polygon
                im_data, text_polys, text_tags = crop_area(im_data, text_polys, text_tags, crop_background=False)
                if text_polys.shape[0] == 0:
                    continue

                h, w, _ = im_data.shape

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im_data.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im_data.copy()
                im_data = im_padded
                # resize the image to input size
                new_h, new_w, _ = im_data.shape
                resize_h = input_size
                resize_w = input_size
                im_data = cv2.resize(im_data, dsize=(resize_w, resize_h))
                resize_ratio_3_x = resize_w/float(new_w)
                resize_ratio_3_y = resize_h/float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im_data.shape
                gt_map = process_label((new_h, new_w), text_polys, text_tags)

                images.append(im_data[:, :, ::-1].astype(np.float32))
                image_files.append(image_file)
                gt_maps.append(gt_map[::cfg.pixel_size, ::cfg.pixel_size, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    yield images, image_files, gt_maps
                    images = []
                    image_files = []
                    gt_maps = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()



if __name__ == '__main__':
    pass
