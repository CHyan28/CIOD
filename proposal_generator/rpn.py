# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List
import torch
import torch.nn.functional as F
from torch import nn

from iod.layers import ShapeSpec
from iod.utils.registry import Registry

from ..anchor_generator import build_anchor_generator
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..distillation_loss import rpn_loss,rpn_adapt_loss
from .build import PROPOSAL_GENERATOR_REGISTRY
from .rpn_outputs import RPNOutputs, find_top_rpn_proposals
from ..spatialtransformnetworks import STN
import random

RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
"""
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.
"""


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    RPN classification and regression heads. Uses a 3x3 conv to produce a shared
    hidden state from which one 1x1 conv predicts objectness logits for each anchor
    and a second 1x1 conv predicts bounding-box deltas specifying how to deform
    each anchor into an object proposal.
    """

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_cell_anchors)) == 1
        ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels, num_cell_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(
            in_channels, num_cell_anchors * box_dim, kernel_size=1, stride=1
        )

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def freeze_layers(self):
        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            for param in l.parameters():
                param.requires_grad = False

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps [B,1024,H,W]
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            # stn = STN(x.size(1),x.size(2),x.size(3))
            # x = stn(x) # [1,1024,30,40] # [1,1024,30,40]
            t = F.relu(self.conv(x)) # torch.Size([2, 1024, 36, 49])
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class RPN(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len        = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features             = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh              = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image    = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction       = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta          = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight             = cfg.MODEL.RPN.LOSS_WEIGHT
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH

        self.anchor_generator = build_anchor_generator(
            cfg, [input_shape[f] for f in self.in_features]
        )
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.anchor_matcher = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        self.rpn_head = build_rpn_head(cfg, [input_shape[f] for f in self.in_features])

        self.base_model = None
        self.enable_rpn_distill = cfg.DISTILL.RPN
        self.freeze_weights = cfg.MODEL.RPN.FREEZE_WEIGHTS

    def set_base_model(self, base_model):
        self.base_model = base_model



    def forward(self, images, features, gt_instances=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances] or None
            loss: dict[Tensor]
        """
        gt_boxes = [x.gt_boxes for x in gt_instances] if gt_instances is not None else None # 新类图像的边界框坐标
        del gt_instances

        if self.freeze_weights:
            self.rpn_head.freeze_layers()
        features = [features[f] for f in self.in_features] # self.in_features = ['res4'] ; features[0].size():torch.Size([2, 1024, 32, 45])
        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features) # pred_objectness_logits: torch.Size([2, 15, 32, 45])  pred_anchor_deltas: torch.Size([2, 60, 32, 45])
        anchors = self.anchor_generator(features) # 每个图像生成[26460,4]个anchors
        # TODO: The anchors only depend on the feature map shape; there's probably
        # an opportunity for some optimizations (e.g., caching anchors).
        outputs = RPNOutputs(
            self.box2box_transform,
            self.anchor_matcher,
            self.batch_size_per_image,
            self.positive_fraction,
            images,
            pred_objectness_logits,
            pred_anchor_deltas,
            anchors,
            self.boundary_threshold,
            gt_boxes,
            self.smooth_l1_beta,
        )

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()} # 求损失的时候开始匹配真实边界框与生成的anchors
            if self.base_model is not None and self.enable_rpn_distill:
                # prev_pred_objectness_logits[0].size(): torch.Size([1, 15, 30, 40])
                # pred_anchor_deltas[0].size(): torch.Size([1, 60, 30, 40])
                prev_pred_objectness_logits, prev_pred_anchor_deltas = self.base_model.proposal_generator.rpn_head(features) # 生成的特征计算rpn预测分数
                rpn_dist_loss = rpn_loss(pred_objectness_logits, pred_anchor_deltas, prev_pred_objectness_logits, prev_pred_anchor_deltas)
                # rpn_dist_loss = rpn_adapt_loss(pred_objectness_logits, pred_anchor_deltas, prev_pred_objectness_logits, prev_pred_anchor_deltas)
                # rpn_output_source = prev_pred_objectness_logits, prev_pred_anchor_deltas
                # rpn_output_target = pred_objectness_logits, pred_anchor_deltas
                # rpn_dist_loss = rpn_norm_loss(rpn_output_source, rpn_output_target, cls_loss='masked_filtered_l2', bbox_loss='l2', bbox_threshold=0.1) # rpn_dist_loss = 0 
                losses.update(rpn_dist_loss)
        else:
            losses = {}

        if self.base_model is not None:
            # Has distillation enabled
            prev_pred_objectness_logits, prev_pred_anchor_deltas = self.base_model.proposal_generator.rpn_head(features)
            # proposals, _ = self.base_model.proposal_generator(images, features)
            # prev_outputs = RPNOutputs(
            #     self.box2box_transform,
            #     self.anchor_matcher,
            #     self.batch_size_per_image,
            #     self.positive_fraction,
            #     images,
            #     prev_pred_objectness_logits,
            #     prev_pred_anchor_deltas,
            #     anchors,
            #     self.boundary_threshold,
            #     None,
            #     self.smooth_l1_beta,
            # )
            # prev_proposals = find_top_rpn_proposals(
            #     prev_outputs.predict_proposals(),
            #     prev_outputs.predict_objectness_logits(),
            #     images,
            #     self.nms_thresh,
            #     self.pre_nms_topk[False],
            #     self.post_nms_topk[False],
            #     self.min_box_side_len,
            #     False,
            # )
            # inds = [p.objectness_logits.sort(descending=True)[1] for p in prev_proposals]
            # index = []
            # for i in range(len(inds)):
            #     index.append(inds[i][:128])
            # prev_proposals = [p[ind] for p, ind in zip(prev_proposals, index)] # torch.Size([128, 4])和torch.Size([128])
            # prev_proposal_boxes = [x.proposal_boxes for x in prev_proposals] # [128,4]
            # prev_boxes = self.base_model.roi_heads._shared_roi_transform(features, prev_proposal_boxes) # torch.Size([1024, 1024, 14, 14])
            # torch.Size([128, 21]) 和 torch.Size([128, 4])
            pass

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            # proposals:寻找rpn得分较高的anchors
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(), 
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )
            # 选择2000个边界框和对应的分数
            # proposals[0]._fields['proposal_boxes']：共预测2000个框 torch.Size([2000])
            # proposals[0]._fields['objectness_logits']：共预测2000个分数  torch.Size([2000])
            # For RPN-only models, the proposals are the final output and we return them in
            # high-to-low confidence order.
            # For end-to-end models, the RPN proposals are an intermediate state
            # and this sorting is actually not needed. But the cost is negligible. 对objectness_logits进行排序
            inds = [p.objectness_logits.sort(descending=True)[1] for p in proposals]
            proposals = [p[ind] for p, ind in zip(proposals, inds)] 
            # proposals[0]._fields['proposal_boxes']：torch.Size([2000, 4]);
            # proposals[0]._fields['objectness_logits'].size(): torch.Size([2000])
        # if self.base_model is not None:
        #     return proposals, prev_boxes,losses
        # else:
        return proposals, losses

