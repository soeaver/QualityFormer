import cv2
import numpy as np
import os
import pycocotools.mask as mask_utils
import random

import torch

from lib.ops import roi_align_rotated

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
TO_REMOVE = 1


class Instance(object):
    def __init__(self, bbox, image_size, labels, ann_types=None, instances=None):
        self.bbox = bbox
        self.size = image_size  # (w, h)
        self.labels = labels
        self.ann_types = ann_types
        self.instances = {}
        self.aspect_ratio = 1.0
        self.trans = None
        # if 'mask' in self.ann_types:
        self.instances['parsing'] = Parsing(instances['parsing'])

    def convert(self, aspect_ratio, scale_ratio):
        """
        (x0, y0, w, h) ==> (xc, yc, w, h, a)
        (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
        """
        x0, y0, w, h = self.bbox[:4]
        xc = x0 + w * 0.5
        yc = y0 + h * 0.5

        if w > aspect_ratio * h:
            h = w * 1.0 / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        w *= 1.25
        h *= 1.25

        self.bbox = torch.tensor([xc, yc, w, h, 0.])
        self.aspect_ratio = aspect_ratio

        if scale_ratio != 1:
            self.bbox = self.bbox * scale_ratio
        for ann_type in self.ann_types:
            self.instances[ann_type].convert(scale_ratio)

    def scale(self, scale_factor):
        s = np.clip(np.random.randn() * scale_factor + 1, 1 - scale_factor, 1 + scale_factor)
        self.bbox[2:4] *= torch.as_tensor(s)

    def rotate(self, rotation_factor):
        r = np.clip(np.random.randn() * rotation_factor, -rotation_factor * 2, rotation_factor * 2) \
            if random.random() <= 0.6 else 0
        self.bbox[4] = torch.as_tensor(r)

    def flip(self):
        # flip center
        self.bbox[0] = self.size[0] - self.bbox[0] - TO_REMOVE
        for ann_type in self.ann_types:
            self.instances[ann_type].flip(self.size[0])

    def crop_and_resize(self, train_size, affine_mode='cv2'):
        self.trans = get_affine_transform(self.bbox, train_size) if affine_mode == 'cv2' else None
        for ann_type in self.ann_types:
            self.instances[ann_type].crop_and_resize(self.bbox, train_size, self.trans)

    def generate_target(self, target_type, sigma, prob_size, train_size):
        target = {}
        if 'parsing' in self.ann_types:
            target['parsing'] = self.instances['parsing'].parsing.long()

        return target

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'type={})'.format(self.type)
        return s


class Parsing(object):
    FLIP_MAP = ()

    def __init__(self, parsing_list):
        root_dir, file_name, parsing_id = parsing_list
        human_dir = root_dir.replace('Images', 'Human_ids')
        category_dir = root_dir.replace('Images', 'Category_ids')
        file_name = file_name.replace('jpg', 'png')
        human_path = os.path.join(human_dir, file_name)
        category_path = os.path.join(category_dir, file_name)
        human_mask = cv2.imread(human_path, 0)
        category_mask = cv2.imread(category_path, 0)
        parsing = category_mask * (human_mask == parsing_id)
        self.parsing = parsing

    def convert(self, scale_ratio):
        if scale_ratio != 1:
            self.parsing = cv2.resize(self.parsing, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_NEAREST)

    def flip(self, image_w=None):
        flipped_parsing = self.parsing[:, ::-1]
        for l_r in Parsing.FLIP_MAP:
            left = np.where(flipped_parsing == l_r[0])
            right = np.where(flipped_parsing == l_r[1])
            flipped_parsing[left] = l_r[1]
            flipped_parsing[right] = l_r[0]

        self.parsing = flipped_parsing

    def crop_and_resize(self, bbox, train_size, trans):
        if trans is None:
            parsing = torch.from_numpy(np.ascontiguousarray(self.parsing)).to(dtype=torch.float32)
            bbox = bbox[None]
            batch_inds = torch.tensor([0.])[None]
            rois = torch.cat([batch_inds, bbox], dim=1)  # Nx5

            self.parsing = roi_align_rotated(
                parsing[None, None], rois, (train_size[1], train_size[0]), 1.0, 1, True, "nearest"
            ).squeeze()
        else:
            parsing = cv2.warpAffine(
                self.parsing,
                trans,
                (int(train_size[0]), int(train_size[1])),
                flags=cv2.INTER_NEAREST
            )
            self.parsing = torch.from_numpy(parsing)


def get_affine_transform(box, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    center = np.array([box[0], box[1]], dtype=np.float32)
    scale = np.array([box[2], box[3]], dtype=np.float32)
    rot = box[4]

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def point_affine(points, bbox, out_size):
    points = np.array(points, dtype=np.float32)
    theta = np.pi * bbox[4] / 180
    cos = np.cos(theta)
    sin = np.sin(theta)

    points[:, 0] = (points[:, 0] - (bbox[0] - bbox[2] / 2)) * out_size[0] / bbox[2]
    points[:, 1] = (points[:, 1] - (bbox[1] - bbox[3] / 2)) * out_size[1] / bbox[3]

    points[:, 0] -= (out_size[0] / 2)
    points[:, 1] -= (out_size[1] / 2)
    x = points[:, 1] * sin + points[:, 0] * cos + out_size[0] / 2
    y = points[:, 1] * cos - points[:, 0] * sin + out_size[1] / 2

    points[:, 0] = x
    points[:, 1] = y

    return points


def mask_to_bbox(mask):
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]
    return [[x0, y0], [x1, y1]]
