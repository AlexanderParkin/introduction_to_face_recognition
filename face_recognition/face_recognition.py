import argparse
import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torchvision

from ir_model import IR_50

class FaceRecognizer(object):
    def __init__(self, device = 'cpu', weights='weights/face_recognition_backbone_ir50_ms1m_epoch120.pth'):
        self.device = device
        self.model = IR_50([112, 112])
        weights = torch.load(weights, map_location='cpu')
        self.model.load_state_dict(weights)
        self.model.eval()
        self.model = self.model.to(self.device)

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((112,112)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def get_descriptor(self, image):
        input_tensor = self.preprocess(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            output = F.normalize(output, dim=1)
            descriptor = output.squeeze().cpu().numpy()

        return descriptor