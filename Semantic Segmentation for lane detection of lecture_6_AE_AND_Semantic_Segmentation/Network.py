from torchvision import models
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.backbone = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21)
		self.backbone.classifier = DeepLabHead(2048, 1)

	def forward(self,x):
		return self.backbone(x)