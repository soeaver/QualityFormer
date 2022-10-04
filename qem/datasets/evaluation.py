import json
import os
import pickle
import shutil
import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.sparse import csr_matrix

from lib.data.evaluation.parsing_eval import ParsingEvaluator, generate_parsing_result
from lib.utils.visualizer import Visualizer

from qem.datasets import dataset_catalog


class Evaluation(object):
    def __init__(self, cfg, training=False):
        """
        Evaluation
        :param cfg: config
        """
        self.cfg = cfg
        self.training = training
        self.iou_types = ()
        self.pet_results = {}
        self.all_iou_types = ("parsing",)

    def parsing_eval(self, iou_type, dataset, output_folder):
        """Interface of Parsing evaluation
        """
        gt_im_dir = dataset_catalog.get_im_dir(self.cfg.TEST.DATASETS[0])
        metrics = self.cfg.PARSING.METRICS if not self.training else ['mIoU', ]
        pet_eval = ParsingEvaluator(
            dataset, self.pet_results[iou_type], gt_im_dir, output_folder, self.cfg.PARSING.SCORE_THRESH,
            self.cfg.PARSING.NUM_PARSING, metrics=metrics
        )
        pet_eval.evaluate()
        pet_eval.accumulate()
        pet_eval.summarize()
        mIoU = pet_eval.stats['mIoU']
        if 'lvis' in self.cfg.TEST.DATASETS[0]:
            pet_eval.print_results()
        return mIoU

    def coco_eval(self, iou_type, dataset, output_folder):
        """Interface of COCO evaluation
        """
        file_path = os.path.join(output_folder, iou_type + ".json")
        pet_eval = evaluate_on_coco(self.cfg, dataset.coco, self.pet_results[iou_type], file_path, iou_type)
        pet_eval.evaluate()
        pet_eval.accumulate()
        pet_eval.summarize()
        mAP = 0.0 if 'lvis' in self.cfg.TEST.DATASETS[0] else pet_eval.stats[0]
        if 'lvis' in self.cfg.TEST.DATASETS[0]:
            pet_eval.print_results()
        return mAP

    def post_processing(self, results, targets, image_ids, dataset):
        """Prepare results by preparing function of each task
        """
        num_im = len(image_ids)
        eval_results = []
        ims_results = []
        prepare_funcs = []
        prepare_funcs = self.get_prepare_func(prepare_funcs)
        for prepare_func in prepare_funcs:
            prepared_result = self.prepare_results(results, targets, image_ids, dataset, prepare_func)
            if prepared_result is not None:
                assert len(prepared_result) >= 2
                eval_results.append(prepared_result[0])
                # box results include box and label
                ims_results.extend(prepared_result[1:])
            else:
                eval_results.append([])
                ims_results.append([None for _ in range(num_im)])
        if self.cfg.VIS.ENABLED:
            self.vis_processing(ims_results, targets, image_ids, dataset)
        return eval_results

    def vis_processing(self, ims_results, targets, image_ids, dataset):
        ims_dets = [
            np.hstack((target.im_bbox.numpy(), target.scores.numpy()[:, np.newaxis])).astype(np.float32, copy=False)
            for target in targets
        ]
        ims_labels = [target.labels.tolist() for target in targets]

        ims_parss = ims_results
        for k, idx in enumerate(image_ids):
            if len(ims_dets[k]) == 0:
                continue

            im = dataset.pull_image(idx)
            visualizer = Visualizer(self.cfg.VIS, im, dataset=dataset)
            im_name = dataset.get_img_info(image_ids[k])['file_name']
            vis_im = visualizer.vis_preds(
                boxes=ims_dets[k],
                classes=ims_labels[k],
                masks=None,
                keypoints=None,
                parsings=ims_parss[k],
                uvs=None,
            )
            cv2.imwrite(os.path.join(self.cfg.CKPT, 'vis', '{}'.format(im_name)), vis_im)

    def evaluation(self, dataset, all_results):
        """Eval results by iou types
        """
        output_folder = os.path.join(self.cfg.CKPT, 'test')
        self.get_pet_results(all_results)

        for iou_type in self.iou_types:
            if iou_type == "parsing":
                eval_result = self.parsing_eval(iou_type, dataset, output_folder)
            elif iou_type in self.all_iou_types:
                eval_result = self.coco_eval(iou_type, dataset, output_folder)
            else:
                raise KeyError("{} is not supported!".format(iou_type))
        if self.cfg.CLEAN_UP:  # clean up all the test files
            shutil.rmtree(output_folder)
        return eval_result

    def prepare_results(self, results, targets, image_ids, dataset, prepare_func=None):
        """Prepare result of each task for evaluation
        """
        if prepare_func is not None:
            return prepare_func(self.cfg, results, targets, image_ids, dataset)
        else:
            return None

    def get_pet_results(self, all_results):
        """Get preparing function of each task
        """
        all_parss = all_results[0]
        if self.cfg.MODEL.PARSING_ON:
            self.iou_types = self.iou_types + ("parsing",)
            self.pet_results['parsing'] = all_parss

    def get_prepare_func(self, prepare_funcs):
        """Get preparing function of each task
        """
        if self.cfg.MODEL.PARSING_ON:
            prepare_funcs.append(prepare_parsing_results)
        else:
            prepare_funcs.append(None)

        return prepare_funcs


def prepare_parsing_results(cfg, results, targets, image_ids, dataset):
    pars_results = []
    ims_parss = []
    output_folder = os.path.join(cfg.CKPT, 'test')

    if 'parsing' not in results.keys():
        return pars_results, ims_parss

    for i, target in enumerate(targets):
        image_id = image_ids[i]
        original_id = dataset.id_to_img_map[image_id]
        if len(target) == 0:
            ims_parss.append(None)
            continue
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        parsings = results['parsing']['ims_parsings'][i][:, :image_height, :image_width]
        # calculating quality scores
        parsing_bbox_scores = target.scores
        parsing_iou_scores = results['parsing']['parsing_iou_scores'][i]
        parsing_instance_pixel_scores = results['parsing']['parsing_instance_pixel_scores'][i]
        parsing_part_pixel_scores = results['parsing']['parsing_part_pixel_scores'][i]
        hcm = results['parsing']['hcm'][i]
        alpha, beta, gamma = cfg.PARSING.QUALITY_WEIGHTS
        instance_dot = torch.pow(parsing_bbox_scores, alpha) * torch.pow(parsing_iou_scores, beta) * \
                       torch.pow(parsing_instance_pixel_scores, gamma)
        instance_scores = torch.pow(instance_dot, 1. / sum((alpha, beta, gamma))).tolist()
        part_dot = torch.stack([torch.pow(parsing_bbox_scores, alpha) * torch.pow(parsing_iou_scores, beta)] *
                               (cfg.PARSING.NUM_PARSING - 1), dim=1) * torch.pow(parsing_part_pixel_scores, gamma)
        part_scores = torch.pow(part_dot, 1. / sum((alpha, beta, gamma))).tolist()
        labels = target.labels.tolist()
        ims_parss.append(parsings.numpy())
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        parsings, instance_scores = generate_parsing_result(
            parsings, instance_scores, part_scores, parsing_bbox_scores.tolist(), semseg=None, img_info=img_info,
            output_folder=output_folder, score_thresh=cfg.PARSING.SCORE_THRESH,
            semseg_thresh=cfg.PARSING.SEMSEG_SCORE_THRESH, parsing_nms_thres=cfg.PARSING.PARSING_NMS_TH,
            num_parsing=cfg.PARSING.NUM_PARSING, hcm=hcm
        )
        pars_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "parsing": csr_matrix(parsing),
                    "score": instance_scores[k]
                }
                for k, parsing in enumerate(parsings)
            ]
        )
    return pars_results, ims_parss
