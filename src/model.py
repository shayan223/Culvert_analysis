from torch import hub, nn


class Model(nn.Module):
    def __init__(self, backbone="detr_resnet101", num_classes=2, num_queries=512):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.backbone = backbone

        self.model = hub.load(
            "facebookresearch/detr",
            self.backbone,
            pretrained=True,
            # num_classes=self.num_classes,
        )
        self.in_features = self.model.class_embed.in_features

        hidden_dim = self.model.transformer.d_model

        self.model.class_embed = nn.Linear(in_features=self.in_features, out_features=self.num_classes)
        self.model.num_queries = self.num_queries
        self.model.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1, groups=4)
        self.model.query_embed = nn.Embedding(num_queries, hidden_dim)