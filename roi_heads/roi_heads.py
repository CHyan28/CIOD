# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Dict
import torch
from torch import nn
import random


from iod.layers import ShapeSpec
from iod.structures import Boxes, Instances, pairwise_iou
# from detectron2.utils.events import get_event_storage
from iod.utils.events import EventStorage, get_event_storage
from iod.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, make_stage
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from ..distillation_loss import roi_head_loss, roi_pooled_feature_loss,attention,roi_stn_feature_loss,roi_head_edge_loss
from .box_head import GCNET, build_box_head
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from iod.structures.boxes import Boxes
from .keypoint_head import build_keypoint_head, keypoint_rcnn_inference, keypoint_rcnn_loss
from .mask_head import build_mask_head, mask_rcnn_inference, mask_rcnn_loss
from ..spatialtransformnetworks import STN, SSTN
import torch.nn.functional as F


ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape, feature_store=None):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape, feature_store)


def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals):
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection).squeeze(1)
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])
    
    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)不匹配的proposals标记为背景类
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1 # 标记忽视的proposals为-1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals) # 把真实边界框加入至proposals中 [2001,4]

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        # gt与proposals继续匹配
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item()) # num_bg_samples = [511,498] 背景类 = 20
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1]) # num_fg_samples = [1,14]
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
            detected instances. Returned during inference only; may be [] during training.

            losses (dict[str->Tensor]):
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape, feature_store=None):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        self.attention            = cfg.MODEL.ROI_BOX_HEAD.NAME  
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.feature_store = feature_store
        self.enable_warp_grad = cfg.WG.ENABLE

        self.res5, out_channels = self._build_res5_block(cfg) # out_channels = 2048
    
        self.box_predictor = FastRCNNOutputLayers(
            out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        self.num_base_class = cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES
        self.num_novel_class = cfg.MODEL.ROI_HEADS.NUM_NOVEL_CLASSES
        self.num_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_HEADS.LEARN_INCREMENTALLY:
            if cfg.MODEL.ROI_HEADS.TRAIN_ON_BASE_CLASSES:
                self.invalid_class_range = list(range(self.num_base_class, self.num_class))
            else:
                self.invalid_class_range = list(range(self.num_base_class + self.num_novel_class, self.num_class))
        else:
            self.invalid_class_range = []
        logging.getLogger(__name__).info("Invalid class range: " + str(self.invalid_class_range))

        self.base_model = None
        self.enable_roi_distillation = cfg.DISTILL.ROI_HEADS
        self.distill_only_fg_roi = cfg.DISTILL.ONLY_FG_ROIS
        # self.dist_loss_weight = cfg.DISTILL.LOSS_WEIGHT
        self.dist_loss_weight = random.random()
        self.enable_distillation = cfg.DISTILL.ENABLE
        self.distill_roi_feature = cfg.DISTILL.ROI_FEATURE
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
        self.finetune = cfg.FINETUNE.ENABLE
       
    def set_base_model(self, base_model):
        self.base_model = base_model

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
         
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return x

    def get_predictions_from_boxes(self, boxes):
        box_features = self.res5(boxes) # torch.Size([512, 2048, 7, 7])
        feature_pooled = box_features.mean(dim=[2, 3]) # torch.Size([512, 2048])
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled) # pred_class_logits: torch.Size([512, 21]); pred_proposal_deltas:torch.Size([512, 4])
        if self.distill_roi_feature:
            return feature_pooled,pred_class_logits,pred_proposal_deltas
        else:
            del feature_pooled
            return pred_class_logits, pred_proposal_deltas


    def get_prediction(self, boxes):
        box_features = self.res5(boxes) # torch.Size([512, 2048, 7, 7])
        feature_pooled = box_features.mean(dim=[2, 3]) # torch.Size([512, 2048])
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled) # pred_class_logits: torch.Size([512, 21]); pred_proposal_deltas:torch.Size([512, 4])
        return feature_pooled,pred_class_logits,pred_proposal_deltas


    def get_warp_loss(self):
        """
        Steps:
            1) Retrieve from features and proposals
            2) Compute the losses
        :return:
        """
        features = []
        proposals = []
        for feats, props in self.feature_store.retrieve():
            features.append(feats)
            proposals.append(props)

        roi_pooled_features = torch.cat(features, dim=0)
        proposals_with_gt = [Instances.cat(proposals, ignore_dim_change=True)]

        pred_class_logits, pred_proposal_deltas = self.get_predictions_from_boxes(roi_pooled_features)
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals_with_gt,
            self.smooth_l1_beta,
            self.invalid_class_range,
            self.dist_loss_weight,
            self.enable_distillation
        )
        losses = outputs.losses()
        losses["loss_cls_warp"] = losses.pop("loss_cls")
        losses["loss_box_reg_warp"] = losses.pop("loss_box_reg")
        return losses

    def update_feature_store(self, features, proposals, targets):
        """
        Feature store (FS) is used to update the warp layers of the ROI Heads. Updating FS involves the following
        Steps:
            1) 'proposals' are filtered per class
            2) The following is done: proposals -> features from BB -> ROI Pooled features
            3) Update the Feature Store
        :param proposals: Proposals from the RPN per image.
        :param features: The backbone feature map.
        :param targets: Ground Truth.
        :return: None; updates self.feature_store.
        """
        proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        # 'boxes' contains the RIO-Pooled features.
        boxes = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        all_proposals = Instances.cat(proposals, ignore_dim_change=True)
        for i in range(len(all_proposals)):
            proposal = all_proposals[i]
            class_id = proposal.gt_classes.item()
            # if class_id != self.num_class:
            self.feature_store.add(((boxes[i].unsqueeze(0).clone().detach(), proposal),), (class_id,))



    def predict_proposals(self, anchors, pred_anchor_deltas):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.
        # 根据anchors和预测的anchor deltas 生成proposals

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
        """
        proposals = []
        # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
        anchors = list(zip(*anchors))
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i[0].tensor.size(1)
            N, _, Hi, Wi = pred_anchor_deltas_i.shape
            # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
            pred_anchor_deltas_i = (
                pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
            )
            # Concatenate all anchors to shape (N*Hi*Wi*A, B)
            # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
            anchors_i = type(anchors_i[0]).cat(anchors_i)
            proposals_i = self.box2box_transform.apply_deltas(
                pred_anchor_deltas_i, anchors_i.tensor
            )
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals


    def calculate_soften_label(self, images, features, proposal_boxes, targets=None):
        del images
        del targets
        soften_boxes = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        soften_box_features = self.res5(soften_boxes) # torch.Size([4096, 2048, 7, 7]) torch.Size([512, 2048, 7, 7])
        soften_feature_pooled = soften_box_features.mean(dim=[2, 3])
        pred_class_logits, _ = self.box_predictor(soften_feature_pooled)
        # soften_feature_pooled_ = torch.split(soften_feature_pooled, split_size_or_sections = soften_feature_pooled.size(0)//2)
        return soften_feature_pooled,pred_class_logits
    
    def forward(self, images, features, proposals, targets=None):
    # def forward(self, images, features, proposals, prev_boxes,targets=None):

        """
        See :class:`ROIHeads.forward`.
        """
        # if self.base_model is not None and self.enable_roi_distillation:
        #     # features = [features[f] for f in self.in_features]
        #     proposals,_ = self.base_model.proposal_generator(images,features)
        #     prev_pred_objectness_logits, prev_pred_anchor_deltas = self.base_model.proposal_generator.rpn_head(features)
        #     anchors = self.base_model.proposal_generator.anchor_generator(features)
        #     selected_proposals, feature_proposals = self.select_proposals(proposals)
        del images

        if self.training:
            # proposals中包含proposal_boxes、objectness_logits、gt_classes和gt_boxes
            # 增加更新特征存储器的代码
            # if self.finetune:
            #     self.update_feature_store(features,proposals, targets)
            #     losses = self.get_warp_loss()
            proposals = self.label_and_sample_proposals(proposals, targets) # 筛选出512个新类的正anchors 格式：Instances(num_instances = 512,image_size,proposals_boxes,)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals] #[512,4] [Boxes(tensor([],[],[]))] proposals[0]._fields['objectness_logits'].size(): torch.Size([512])
        # 'boxes' contains the RIO-Pooled features.
        boxes = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        ) # self.in_features : ['res4']  torch.Size([4096, 1024, 14, 14]) torch.Size([512, 1024, 14, 14]); torch.Size([1024, 1024, 14, 14])

        box_features = self.res5(boxes) # torch.Size([4096, 2048, 7, 7]) torch.Size([512, 2048, 7, 7]); torch.Size([1024, 2048, 7, 7])
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1 torch.Size([4096, 2048]) torch.Size([1024, 2048])
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled) # 所有类别的roi预测输出
        if self.distill_roi_feature == "False":
            del feature_pooled
        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.invalid_class_range,
            self.dist_loss_weight,
            self.enable_distillation
        )

        if self.training:
            losses = outputs.losses() # losses['loss_cls']=tensor(2.6288, device='cuda:0');losses['loss_box_reg'] = tensor(0.0490, device='cuda:0')
            if self.base_model is not None and self.enable_roi_distillation:
                # 蒸馏前景RoIs self.distill_only_fg_roi = False 蒸馏所有的前景框
                if self.distill_only_fg_roi:
                    proposals_fg = [p[p.gt_classes != 20] for p in proposals]
                    proposal_boxes_fg = [x.proposal_boxes for x in proposals_fg]
                    boxes_fg = self._shared_roi_transform(
                        [features[f] for f in self.in_features], proposal_boxes_fg
                    )
                    pred_class_logits, pred_proposal_deltas = self.get_predictions_from_boxes(boxes_fg)
                    prev_pred_class_logits, prev_pred_proposal_deltas = self.base_model.roi_heads.\
                        get_predictions_from_boxes(boxes_fg)
                    # roi_dist_loss = roi_head_loss(pred_class_logits[:, 0:self.num_base_class], pred_proposal_deltas,
                    #                             prev_pred_class_logits[:, 0:self.num_base_class],
                    #                             prev_pred_proposal_deltas, self.dist_loss_weight) # RoI输出蒸馏：所有类中对基类的预测输出求损失
                    # losses.update(roi_dist_loss)
                else:
                    # self.distill_roi_feature = False
                    if self.distill_roi_feature:
                        _,prev_pred_class_logits, prev_pred_proposal_deltas = self.base_model.roi_heads.\
                            get_predictions_from_boxes(boxes)
                    else:
                        prev_pred_class_logits, prev_pred_proposal_deltas = self.base_model.roi_heads.\
                            get_predictions_from_boxes(boxes)  # boxes可以替换成是旧模型输出的boxes；prev_pred_class_logits的计算是通过新类经gt选择之后的boxes计算的
                        # prev_box_features = self.base_model.roi_heads.res5(boxes)
                        # prev_feature_pooled,prev_pred_class_logits, prev_pred_proposal_deltas = self.base_model.roi_heads.\
                        #     self.base_model.roi_heads.get_prediction(boxes)
                        # prev_feature_pooled = self.base_model.roi_heads.res5(boxes).mean(dim=[2, 3])
                # if self.distill_roi_feature:
                #         roi_dist_loss = roi_head_loss(pred_class_logits[:, 0:self.num_base_class], pred_proposal_deltas,
                #                                 prev_pred_class_logits[:, 0:self.num_base_class],
                #                                 prev_pred_proposal_deltas, self.dist_loss_weight)       

                #         prev_box_features = self.base_model.roi_heads.res5(prev_boxes)
                #         # prev_box_features = self.base_model.roi_heads.res5(boxes)
                #         prev_pooled_features = prev_box_features.mean(dim=[2, 3])  # pooled to 1x1 torch.Size([4096, 2048]) torch.Size([1024, 2048])
                #         prev_pred_class_logits, prev_pred_proposal_deltas = self.base_model.roi_heads.box_predictor(prev_pooled_features)
                #         assert prev_pred_class_logits.size(1) == 21
                #         # prev_fea_att = self.sstn_fea(prev_box_features)
                #         del prev_pooled_features 
                #         # prev_pred_class_logits, prev_pred_proposal_deltas = self.base_model.roi_heads.\
                #         #     get_predictions_from_boxes(boxes)
                #         dim = prev_pred_class_logits.size(0)
                #         roi_dist_loss = roi_head_loss(pred_class_logits[:dim, 0:self.num_base_class], pred_proposal_deltas[:dim,:],
                #                                     prev_pred_class_logits[:, 0:self.num_base_class],
                #                                     prev_pred_proposal_deltas, self.dist_loss_weight)
                    # else:
                        # proposals_bg = [p[p.gt_classes == 20] for p in proposals]  # 蒸馏学生模型标注的背景边界框，其中包括老师模型预测的前景框
                        # proposal_boxes_bg = [x.proposal_boxes for x in proposals_bg]
                        # boxes_bg = self._shared_roi_transform(
                        #     [features[f] for f in self.in_features], proposal_boxes_bg
                        # )
                        # pred_class_logits, pred_proposal_deltas = self.get_predictions_from_boxes(boxes_bg)
                        # prev_pred_class_logits, prev_pred_proposal_deltas = self.base_model.roi_heads.\
                        #     get_predictions_from_boxes(boxes_bg)
                    roi_dist_loss = roi_head_loss(pred_class_logits[:, 0:self.num_base_class], pred_proposal_deltas,
                                            prev_pred_class_logits[:, 0:self.num_base_class],
                                            prev_pred_proposal_deltas, self.dist_loss_weight) # RoI输出蒸馏：所有类中对基类的预测输出求损失; 从边界框入手，老师模型对背景边界框的预测和学生模型对背景边界框的预测求损失
                    losses.update(roi_dist_loss)
                        # roi_dist_loss = roi_head_edge_loss(pred_class_logits[:, 0:self.num_base_class], pred_proposal_deltas,
                        #                       prev_pred_class_logits[:, 0:self.num_base_class],
                        #                       prev_pred_proposal_deltas, prev_box_features,box_features,self.dist_loss_weight)
                        # losses.update(roi_dist_loss)
            del features
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = mask_rcnn_loss(mask_logits, proposals)
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            mask_logits = self.mask_head(x)
            mask_rcnn_inference(mask_logits, instances)
        return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape, feature_store=None):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)
        self._init_mask_head(cfg)
        self._init_keypoint_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

        self.num_base_class = cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES
        self.num_novel_class = cfg.MODEL.ROI_HEADS.NUM_NOVEL_CLASSES
        self.num_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        if cfg.MODEL.ROI_HEADS.LEARN_INCREMENTALLY:
            if cfg.MODEL.ROI_HEADS.TRAIN_ON_BASE_CLASSES:
                self.invalid_class_range = list(range(self.num_base_class, self.num_class))
            else:
                self.invalid_class_range = list(range(self.num_base_class + self.num_novel_class, self.num_class))
        else:
            self.invalid_class_range = []
        logging.getLogger(__name__).info("Invalid class range: " + str(self.invalid_class_range))

        self.base_model = None
        self.enable_roi_distillation = cfg.DISTILL.ROI_HEADS
        self.distill_only_fg_roi = cfg.DISTILL.ONLY_FG_ROIS
        # self.dist_loss_weight = cfg.DISTILL.LOSS_WEIGHT
        self.dist_loss_weight = random.random()
        self.enable_distillation = cfg.DISTILL.ENABLE
    
    def set_base_model(self, base_model):
        self.base_model = base_model


    def get_predictions_from_boxes(self, box_features):
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features
        return pred_class_logits, pred_proposal_deltas
    
    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg):
        # fmt: off
        self.keypoint_on                         = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution                        = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales                            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # noqa
        sampling_ratio                           = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type                              = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            # losses.update(self._forward_mask(features_list, proposals))
            # losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_pool_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_pool_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.invalid_class_range,
        )
        # if self.training:
        #     return outputs.losses()

        if self.training:
            losses = outputs.losses()
            if self.base_model is not None and self.enable_roi_distillation:
                if self.distill_only_fg_roi:
                    proposals_fg = [p[p.gt_classes != 20] for p in proposals]
                    proposal_boxes_fg = [x.proposal_boxes for x in proposals_fg]
                    boxes_fg_pooled = self.box_pooler(
                        [features[f] for f in self.in_features], proposal_boxes_fg
                    )
                    pred_class_logits, pred_proposal_deltas = self.get_predictions_from_boxes(boxes_fg_pooled)
                    prev_pred_class_logits, prev_pred_proposal_deltas = self.base_model.roi_heads.\
                        get_predictions_from_boxes(boxes_fg_pooled)
                else:
                    prev_pred_class_logits, prev_pred_proposal_deltas = self.base_model.roi_heads.\
                        get_predictions_from_boxes(box_pool_features)
                
                roi_dist_loss = roi_head_loss(pred_class_logits[:, 0:self.num_base_class], pred_proposal_deltas,
                                              prev_pred_class_logits[:, 0:self.num_base_class],
                                              prev_pred_proposal_deltas, self.dist_loss_weight)
                losses.update(roi_dist_loss)   

            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                mask_logits = self.mask_head(mask_features)
                losses["loss_mask"] = mask_rcnn_loss(mask_logits, proposals)
            return losses         
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances




    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with iOD fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_head(mask_features)
            return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_keypoint(self, features, instances):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with iOD fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)

            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances
