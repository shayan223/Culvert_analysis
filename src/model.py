import warnings

import torch
from torch import hub, nn
import torch.nn.functional as F

from src.config import DEVICE
from src.utils import box_ops


class Model(nn.Module):
    def __init__(
        self,
        backbone="detr_resnet101",
        num_classes=2,
        num_queries=512,
        pretrained=False,
    ):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.pretraind = pretrained
        self.backbone = backbone

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.model = hub.load(
                "facebookresearch/detr",
                self.backbone,
                pretrained=True,
            )

        self.model.num_classes = self.num_classes
        self.model.to(DEVICE)

        self.in_features = self.model.class_embed.in_features

        hidden_dim = self.model.transformer.d_model

        self.model.class_embed = nn.Linear(hidden_dim, self.num_classes)
        self.model.num_queries = self.num_queries
        self.model.input_proj = nn.Conv2d(
            2048,
            hidden_dim,
            kernel_size=1,
            groups=4,
        )
        self.model.query_embed = nn.Embedding(self.num_queries, hidden_dim)

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
