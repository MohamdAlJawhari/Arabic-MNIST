import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def build_mobilenet(num_classes: int, pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
    else:
        model = mobilenet_v3_small(weights=None)

    # Modify first conv to accept 1 channel instead of 3
    first_conv: nn.Conv2d = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )

    # If pretrained, initialize new_conv from old weights by averaging RGB channels
    with torch.no_grad():
        if pretrained:
            new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

    model.features[0][0] = new_conv

    # Replace classifier head for our number of classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model
