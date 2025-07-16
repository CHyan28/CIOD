# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn

from iod.structures import ImageList
from iod.utils.events import get_event_storage,EventStorage
from iod.utils.logger import log_first_n
from iod.utils.visualizer import Visualizer
from iod.utils.store import Store
import random

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from ..distillation_loss import backbone_loss,backbone_att_loss,spatial_location_feature_distillation,feature_stn_distillation,\
feature_att_distillation,feature_stn_ft_distillation,roi_feature_distillation,roi_graph_distillation,roi_head_loss,\
roi_edge_distillation,feature_stn_vid_distillation,logit_distillation,feature_FactorTransfer_distillation,feature_AT_distillation,feature_SP_distillation
# feature_channel_adaptive_distillation
from .build import META_ARCH_REGISTRY
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from  torchvision import utils as vutils
from ..show_feature_map import show_feature_map
import torch.nn.functional as F




__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.feature_store = Store(cfg.MODEL.ROI_HEADS.NUM_CLASSES+1,
                                   cfg.WG.NUM_FEATURES_PER_CLASS) if cfg.WG.ENABLE else None
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape(), self.feature_store)
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT
        self.base_model = None

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)
        self.enable_backbone_distillation = cfg.DISTILL.BACKBONE # 第一阶段是：True
        self.enable_warp_grad = cfg.WG.ENABLE
        self.cfg = cfg
        self.enable_rpn_distill = cfg.DISTILL.RPN

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.distill_roi_graph_feature = cfg.DISTILL.ROI_GRAPH_FEATURE
        self.num_base_class = cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES
        self.num_novel_class = cfg.MODEL.ROI_HEADS.NUM_NOVEL_CLASSES
        self.num_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.dist_loss_weight = cfg.DISTILL.LOSS_WEIGHT

    def set_base_model(self, base_model):
        self.base_model = base_model
        self.proposal_generator.set_base_model(base_model)
        self.roi_heads.set_base_model(base_model)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """

        inputs = [x for x in batched_inputs]
        prop_boxes = [p for p in proposals]
        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(inputs, prop_boxes):
            img = input["image"].cpu().numpy()
            assert img.shape[0] == 3, "Images should have 3 channels."
            if self.input_format == "BGR":
                img = img[::-1, :, :]
            img = img.transpose(1, 2, 0)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = " 1. GT bounding boxes  2. Predicted proposals"
            storage.put_image(vis_name, vis_img)

    def get_warp_loss(self, all_images_in_store):
        """

        :param batched_inputs: Contains all the images in the Image Store
        :return:
        """
        for image in all_images_in_store:
            img = self.preprocess_image([image])
            if 'instances' in image:
                gt_instances = [image['instances'].to(self.device)]
            elif 'targets' in image:
                gt_instances = [image['targets'].to(self.device)]
            else:
                gt_instances = None

            features = self.backbone(img.tensor)
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(img, features, gt_instances)
            else:
                assert "proposals" in all_images_in_store[0]
                proposals = [x["proposals"].to(self.device) for x in [image]]

            # self.roi_heads.update_feature_store(features, proposals, gt_instances)
            _, detector_losses = self.roi_heads(img, features, proposals, gt_instances)
        # detector_losses = self.roi_heads.get_warp_loss()
        # self.feature_store.reset()
        losses = {}
        losses.update(detector_losses)
        return losses

    # def get_warp_loss(self, all_images_in_store):
    #     """

    #     :param batched_inputs: Contains all the images in the Image Store
    #     :return:
    #     """
    #     for image in all_images_in_store:
    #         img = self.preprocess_image([image])

    #         if 'instances' in image:
    #             gt_instances = [image['instances'].to(self.device)]
    #         elif 'targets' in image:
    #             gt_instances = [image['targets'].to(self.device)]
    #         else:
    #             gt_instances = None

    #         features = self.backbone(img.tensor)
    #         if self.proposal_generator:
    #             proposals, prev_boxes, _ = self.proposal_generator(img, features, gt_instances)
    #         else:
    #             assert "proposals" in all_images_in_store[0]
    #             proposals = [x["proposals"].to(self.device) for x in [image]]

    #         # proposals, _ = self.proposal_generator(img, features, gt_instances)
    #         # self.roi_heads.update_feature_store(features, proposals, gt_instances)
    #         _, detector_losses = self.roi_heads(img, features, proposals, prev_boxes, gt_instances)

    #     # warp_losses = self.roi_heads.get_warp_loss()
    #     self.feature_store.reset()

    #     losses = {}
    #     losses.update(detector_losses)
    #     return losses


    # def forward(self, batched_inputs):
    #     """
    #     Args:
    #         batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
    #             Each item in the list contains the inputs for one image.
    #             For now, each item in the list is a dict that contains:

    #             * image: Tensor, image in (C, H, W) format.
    #             * instances (optional): groundtruth :class:`Instances`
    #             * proposals (optional): :class:`Instances`, precomputed proposals.

    #             Other information that's included in the original dicts, such as:

    #             * "height", "width" (int): the output resolution of the model, used in inference.
    #                 See :meth:`postprocess` for details.

    #     Returns:
    #         list[dict]:
    #             Each dict is the output for one input image.
    #             The dict contains one key "instances" whose value is a :class:`Instances`.
    #             The :class:`Instances` object has the following keys:
    #                 "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
    #     """
    #     if not self.training:
    #         return self.inference(batched_inputs)
    #     # 5_p_5.yaml: self.cfg.WG.TRAIN_WARP = False ; self.cfg.WG.USE_FEATURE_STORE = True
    #     if self.cfg.WG.TRAIN_WARP and self.cfg.WG.USE_FEATURE_STORE:
    #         return self.get_warp_loss(batched_inputs)

    #     images = self.preprocess_image(batched_inputs) # images.tensor: torch.Size([B, 3, 480, 640])
        
    #     if "instances" in batched_inputs[0]:
    #         gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
    #     elif "targets" in batched_inputs[0]:
    #         log_first_n(
    #             logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
    #         )
    #         gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
    #     else:
    #         gt_instances = None
    #     # img = images[0]
    #     # vutils.save_image(img, '/root/data/iOD/visualization/images/1.jpg', normalize=True)
    #     # del img
    #     features = self.backbone(images.tensor) # features['res4'].size() : torch.Size([2, 1024, 32, 45]) [B,C,H,W]
    #     # show_feature_map(features['res4'][0],type="original" )
    #     backbone_dist_loss = 0
    #     # 允许backbone蒸馏
    #     if self.base_model is not None and self.enable_backbone_distillation:
    #         prev_features = self.base_model.backbone(images.tensor) # torch.Size([1, 1024, 30, 40])
    #         if self.cfg.MODEL.ROI_BOX_HEAD.NAME == 'GCNET':
    #             # backbone_dist_loss = backbone_att_loss(features,prev_features)
    #             backbone_dist_loss = spatial_location_feature_distillation(prev_features['res4'],features['res4'],loss='normalized_filtered_l2')
    #             # backbone_dist_loss = feature_stn_ft_distillation(prev_features['res4'],features['res4'],loss='normalized_filtered_l2')
    #             # backbone_dist_loss = feature_stn_distillation(prev_features['res4'],features['res4'],loss='normalized_filtered_l2')
    #             # backbone_dist_loss = feature_att_distillation(prev_features['res4'],features['res4'],loss='normalized_filtered_l2')
    #             # backbone_dist_loss = feature_channel_adaptive_distillation(features['res4'], prev_features['res4'])
    #         else:
    #             backbone_dist_loss = backbone_loss(features, prev_features)
    #     if self.proposal_generator:
    #         # if self.base_model is not None:
    #         #     proposals, prev_boxes, proposal_losses = self.proposal_generator(images, features, gt_instances) 
    #         # else:
    #         proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
    #         # proposals, proposal_losses = self.base_model.proposal_generator(images, features, gt_instances) 
    #         # images和gt_instances仅包含新类，每个图像选择2000个实例框；proposal_losses分为：loss_rpn_cls 和 loss_rpn_loc
    #     else:
    #         assert "proposals" in batched_inputs[0]
    #         proposals = [x["proposals"].to(self.device) for x in batched_inputs]
    #         proposal_losses = {}
    #     # if self.base_model is not None:
    #     #     _, detector_losses = self.roi_heads(images, features, proposals, prev_boxes, gt_instances)
    #     # else:
    #     _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
    #     # detector_losses['loss_cls']: tensor(0.4490)
    #     # detector_losses['loss_box_reg']: tensor(0.1876)
    #     # detector_losses['loss_dist_roi_head']: tensor(1.6704e-07)
    #     if self.vis_period > 0:           
    #         storage = get_event_storage()
    #         if storage.iter % self.vis_period == 0:
    #             self.visualize_training(batched_inputs, proposals)

    #     losses = {}
    #     losses.update(detector_losses)
    #     losses.update(proposal_losses)

    #     if self.base_model is not None and self.enable_backbone_distillation:
    #         losses.update(backbone_dist_loss)

    #     return losses

    def forward(self, batched_inputs):
        """

        """
        if not self.training:
            return self.inference(batched_inputs)
        # 5_p_5.yaml: self.cfg.WG.TRAIN_WARP = False ; self.cfg.WG.USE_FEATURE_STORE = True
        if self.cfg.WG.TRAIN_WARP and self.cfg.WG.USE_FEATURE_STORE:
            return self.get_warp_loss(batched_inputs)

        images = self.preprocess_image(batched_inputs) # images.tensor: torch.Size([B, 3, 480, 640])
        
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        # img = images[0]
        # vutils.save_image(img, '/root/data/iOD/visualization/images/1.jpg', normalize=True)
        # del img
        features = self.backbone(images.tensor) # features['res4'].size() : torch.Size([2, 1024, 32, 45]) [B,C,H,W]
        # show_feature_map(features['res4'][0],type="original" )
        backbone_dist_loss = 0
        # 允许backbone蒸馏
        if self.base_model is not None and self.enable_backbone_distillation:
            prev_features = self.base_model.backbone(images.tensor) # torch.Size([1, 1024, 30, 40])
            if self.cfg.MODEL.ROI_BOX_HEAD.NAME == 'GCNET':
                # backbone_dist_loss = backbone_att_loss(features,prev_features)
                backbone_dist_loss = spatial_location_feature_distillation(prev_features['res4'],features['res4'],loss='normalized_filtered_l2')
                # backbone_dist_loss = feature_stn_vid_distillation(prev_features['res4'],features['res4'],loss='normalized_filtered_l2')
                # backbone_dist_loss = feature_stn_ft_distillation(prev_features['res4'],features['res4'],loss='normalized_filtered_l2')
                # backbone_dist_loss = feature_stn_distillation(prev_features['res4'],features['res4'],loss='normalized_filtered_l2')
                # backbone_dist_loss = feature_att_distillation(prev_features['res4'],features['res4'],loss='normalized_filtered_l2')
                # backbone_dist_loss = feature_FactorTransfer_distillation(features['res4'], prev_features['res4'])
            else:
                backbone_dist_loss = backbone_loss(features, prev_features)
        # 允许roi特征进行图蒸馏
        # if self.base_model is not None and self.distill_roi_graph_feature:
        #     with torch.no_grad():
        #         # 原始的图节点+图的边蒸馏损失
        #         self.base_model.eval()
        #         prev_proposals, _ = self.base_model.proposal_generator(images, prev_features)
        #         old_selected_proposals = self.selected_proposals(prev_proposals)
        #         old_proposal_boxes = [x.proposal_boxes for x in old_selected_proposals]  
        #         old_boxes = self.base_model.roi_heads._shared_roi_transform(
        #             [prev_features[f] for f in self.in_features], old_proposal_boxes
        #         ) 
        #         old_box_features = self.base_model.roi_heads.res5(old_boxes)
        #         pred_old_scores,_ = self.base_model.roi_heads.box_predictor(old_box_features.mean(dim=[2, 3]))
                # old_feature_ = torch.split(old_box_features, split_size_or_sections = old_box_features.size(0)//2)
                # _,pred_class_logits = self.roi_heads.calculate_soften_label(images,features,old_proposal_boxes)
                # sofen_old_feature_ = torch.split(soften_box_features, split_size_or_sections = soften_box_features.size(0)//2)
                # roi_feature_dist_loss = roi_feature_distillation(old_feature_pooled_,sofen_old_feature_pooled_)
                # roi_edge_dist_loss = roi_edge_distillation(soften_box_features,old_box_features)
                # roi_graph_distillation_loss = roi_graph_distillation(roi_feature_dist_loss,roi_edge_dist_loss)
                # loss_logits = F.mse_loss(pred_old_scores[:, 0:self.num_base_class],pred_class_logits[:, 0:self.num_base_class])
                # roi_edge_dist_loss = roi_edge_distillation(pred_old_scores,pred_class_logits)
                # roi_graph_distillation_loss = roi_graph_distillation(roi_edge_dist_loss,loss_logits)

                # self.base_model.eval()
                # outputs = self.base_model(batched_inputs)
                # pred_old_proposals = []
                # for k in range(len(outputs)):
                #     pred_proposals = outputs[k]['instances']
                #     pred_old_proposals.append(pred_proposals)
                # pred_old_boxes = [x.pred_boxes for x in pred_old_proposals]
                # # pred_old_scores = [x.scores for x in pred_old_proposals]
                # # pred_old_classes = [x.pred_classes for x in pred_old_proposals]
                # old_boxes = self.base_model.roi_heads._shared_roi_transform(
                #     [prev_features[f] for f in self.in_features], pred_old_boxes
                # ) 
                # old_box_features = self.base_model.roi_heads.res5(old_boxes)
                # pred_old_scores,_ = self.base_model.roi_heads.box_predictor(old_box_features.mean(dim=[2, 3]))
                # old_feature_pooled = torch.split(old_box_features.mean(dim=[2, 3]), split_size_or_sections = \
                #                                   [pred_old_boxes[0].tensor.size(0),pred_old_boxes[1].tensor.size(0)])
                # soften_feature_pooled,pred_class_logits = self.roi_heads.calculate_soften_label(images,features,pred_old_boxes)
                # soften_feature_pooled_ = torch.split(soften_feature_pooled, split_size_or_sections = \
                #                                   [pred_old_boxes[0].tensor.size(0),pred_old_boxes[1].tensor.size(0)])
                # loss_logits = F.mse_loss(pred_old_scores[:, 0:self.num_base_class],pred_class_logits[:, 0:self.num_base_class])
                # roi_edge_dist_loss = roi_edge_distillation(pred_old_scores,pred_class_logits)
                # # roi_feature_dist_loss = roi_feature_distillation(old_feature_pooled,soften_feature_pooled_)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               roi_edge_dist_loss = roi_edge_distillation(pred_old_scores,pred_class_logits)
                # roi_graph_distillation_loss = roi_graph_distillation(roi_edge_dist_loss,loss_logits)
                # prev_proposals, _ = self.base_model.proposal_generator(images, prev_features)
                # old_selected_proposals = self.selected_proposals(prev_proposals)
                # old_proposal_boxes = [x.proposal_boxes for x in old_selected_proposals]  
                # old_boxes = self.base_model.roi_heads._shared_roi_transform(
                #     [prev_features[f] for f in self.in_features], old_proposal_boxes
                # ) 
                # old_box_features_ = self.base_model.roi_heads.res5(old_boxes)
                # old_features_pooled = old_box_features_.mean(dim=[2, 3])
                # old_features_pooled_ = torch.split(old_features_pooled, split_size_or_sections = old_feature_pooled.size(0)//2)
                # sofen_old_feature_pooled_ = self.roi_heads.calculate_soften_label(images,features,old_proposal_boxes)
                # roi_edge_dist_loss2 = roi_feature_edge_distillation(old_features_pooled_,sofen_old_feature_pooled_)
                # roi_edge_dist_loss = roi_edge_distillation(roi_edge_dist_loss1,roi_edge_dist_loss2)
        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
   
        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        # detector_losses['loss_cls']: tensor(0.4490)
        # detector_losses['loss_box_reg']: tensor(0.1876)
        # detector_losses['loss_dist_roi_head']: tensor(1.6704e-07)
        if self.vis_period > 0:           
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.base_model is not None and self.enable_backbone_distillation:
            losses.update(backbone_dist_loss)
        # if self.base_model is not None and self.distill_roi_graph_feature:
            # losses.update(roi_graph_distillation_loss)
            # losses.update(roi_edge_dist_loss)
        return losses


    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results
        

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        # img = images[0]
        # img = transforms.ToPILImage(img)
        # plt.imshow(img)
        # plt.savefig('/root/data/iOD/visualization/imgs/1.png')
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
    

    def selected_proposals(self, prev_proposals):
        old_selected_proposals = []
        for k in range(len(prev_proposals)):
            inds = [prev_proposals[k]._fields['objectness_logits'].sort(descending=True)[1]]
            # old_proposals = prev_proposals[k][:128]
            old_proposals = prev_proposals[k][inds]
            num_proposals = len(old_proposals)
            if num_proposals < 64:
                list = range(0, num_proposals, 1)
                selected_proposal_index = random.sample(list, num_proposals)
            elif num_proposals < 128:
                    list = range(0, num_proposals, 1)
                    selected_proposal_index = random.sample(list, 64)
            else:
                list = range(0, 128, 1)
                selected_proposal_index = random.sample(list, 64)
            old_proposals = prev_proposals[k][selected_proposal_index]
            old_selected_proposals.append(old_proposals)
        return old_selected_proposals

@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
