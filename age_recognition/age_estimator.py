import argparse
import os
import pandas as pd
import numpy as np

import torch
import torchvision

from ir_model import IR_50

class AgeEstimator(object):
    def __init__(self, device = 'cpu', weights='ir_50_model/test_weights.pth'):
        self.device = device
        self.model = IR_50((112, 112), 1)
        weights = torch.load(weights, map_location='cpu')
        self.model.load_state_dict(weights)
        self.model.eval()
        self.model = self.model.to(self.device)

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((112,112)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def estimate_img(self, image):
        input_tensor = self.preprocess(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            print(output)
            predicted_age = output.squeeze().cpu().numpy()

        return predicted_age