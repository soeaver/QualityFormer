import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from collections import defaultdict

import lib.utils.colormap as colormap_utils
from lib.utils.timer import Timer


class Visualizer(object):
    def __init__(self, cfg, im, dataset=None):

        self.cfg = cfg
        self.im = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])  # BGR255
        self.dataset = dataset

        self._GRAY = [218, 227, 218]
        self._GREEN = [18, 127, 15]
        self._WHITE = [255, 255, 255]

    def vis_preds(self, boxes=None, classes=None, masks=None, keypoints=None, parsings=None, uvs=None, hiers=None,
                  semsegs=None, panos=None, panos_label=None):
        """Constructs a numpy array with the detections visualized."""
        timers = defaultdict(Timer)
        timers['bbox_prproc'].tic()

        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.cfg.VIS_TH:
            return self.im

        # get color map
        ins_colormap = self.get_colormap(self.cfg.SHOW_BOX.COLORMAP)

        # Display in largest to the smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        timers['bbox_prproc'].toc()

        instance_id = 1
        for i in sorted_inds:
            bbox = boxes[i, :-1]
            score = boxes[i, -1]
            if score < self.cfg.VIS_TH:
                continue

            # get instance color (box, class_bg)
            if self.cfg.SHOW_BOX.COLOR_SCHEME == 'category':
                ins_color = ins_colormap[classes[i]]
            elif self.cfg.SHOW_BOX.COLOR_SCHEME == 'instance':
                instance_id = instance_id % len(ins_colormap.keys())
                ins_color = ins_colormap[instance_id]
            else:
                ins_color = self._GREEN
            instance_id += 1

            # show box (on by default)
            if self.cfg.SHOW_BOX.ENABLED:
                if len(bbox) == 4:
                    timers['show_box'].tic()
                    self.vis_bbox((bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), ins_color)
                    timers['show_box'].toc()
                else:
                    raise RuntimeError("Check the box format")

            # show class (on by default)
            if self.cfg.SHOW_CLASS.ENABLED:
                timers['show_class'].tic()
                class_str = self.get_class_string(classes[i], score)
                self.vis_class((bbox[0], bbox[1] - 2), class_str, ins_color)
                timers['show_class'].toc()

            show_parss = True if parsings is not None and self.cfg.SHOW_PARSS.ENABLED and len(parsings) > i else False
            # show parsings
            if show_parss:
                timers['show_parss'].tic()
                parss_colormap = self.get_colormap(self.cfg.SHOW_PARSS.COLORMAP)
                self.vis_parsing(parsings[i][0], parss_colormap, show_masks=False)
                timers['show_parss'].toc()

        # for k, v in timers.items():
        #     print(' | {}: {:.3f}s'.format(k, v.total_time))

        return self.im

    def get_class_string(self, class_index, score=None):
        class_text = self.dataset.classes[class_index] if self.dataset is not None else 'id{:d}'.format(class_index)
        if score is None:
            return class_text
        return class_text + ' {:0.2f}'.format(score).lstrip('0')

    def get_colormap(self, colormap_type, rgb=False):
        colormap = eval('colormap_utils.{}'.format(colormap_type))
        if rgb:
            colormap = colormap_utils.dict_bgr2rgb(colormap)
        return colormap

    def vis_bbox(self, bbox, bbox_color):
        """Visualizes a bounding box."""
        (x0, y0, w, h) = bbox
        x1, y1 = int(x0 + w), int(y0 + h)
        x0, y0 = int(x0), int(y0)
        cv2.rectangle(self.im, (x0, y0), (x1, y1), bbox_color, thickness=self.cfg.SHOW_BOX.BORDER_THICK)

    def vis_class(self, pos, class_str, bg_color):
        """Visualizes the class."""
        font_color = self.cfg.SHOW_CLASS.COLOR
        font_scale = self.cfg.SHOW_CLASS.FONT_SCALE

        x0, y0 = int(pos[0]), int(pos[1])
        # Compute text size.
        txt = class_str
        font = cv2.FONT_HERSHEY_SIMPLEX
        ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
        # Place text background.
        back_tl = x0, y0 - int(1.3 * txt_h)
        back_br = x0 + txt_w, y0
        cv2.rectangle(self.im, back_tl, back_br, bg_color, -1)
        # Show text.
        txt_tl = x0, y0 - int(0.3 * txt_h)
        cv2.putText(self.im, txt, txt_tl, font, font_scale, font_color, lineType=cv2.LINE_AA)

    def vis_parsing(self, parsing, colormap, show_masks=True):
        """Visualizes a single binary parsing."""
        self.im = self.im.astype(np.float32)
        idx = np.nonzero(parsing)

        parsing_alpha = self.cfg.SHOW_PARSS.PARSING_ALPHA
        colormap = colormap_utils.dict2array(colormap)
        parsing_color = colormap[parsing.astype(np.int)]

        border_color = self.cfg.SHOW_PARSS.BORDER_COLOR
        border_thick = self.cfg.SHOW_PARSS.BORDER_THICK

        self.im[idx[0] - 1, idx[1] - 1, :] *= 1.0 - parsing_alpha
        self.im += parsing_alpha * parsing_color

        if show_masks:
            try:
                _, contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            except:
                contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(self.im, contours, -1, border_color, border_thick, cv2.LINE_AA)

        self.im.astype(np.uint8)
