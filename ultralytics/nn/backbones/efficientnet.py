class EfficientNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        efficientnet = efficientnet_b0(weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            efficientnet,
            return_nodes={"features.4": "P3", "features.6": "P4", "features.8": "P5"},
        )

    def forward(self, x):
        out = self.backbone(x)
        return list(out.values())
