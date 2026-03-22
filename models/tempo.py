from lifelines.utils import concordance_index
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import models
import torch.nn as nn
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import shap
from datetime import datetime, timedelta
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# === MRI Net
class MRI_Net(nn.Module):
    def __init__(self, slice_intake=5, dim_output=16):
        super().__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

        checkpoint = torch.load("dataset/MRI_pretrained.pth")
        state_dict = checkpoint['model_state_dict']

        self.model.load_state_dict(state_dict, strict=False)

        self.feature_extractor = nn.Sequential(
            *list(self.model.children())[:-1],  # 移除fc层
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        self.time_fusion = nn.Sequential(
            nn.Linear(512 * slice_intake, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, dim_output),
            nn.BatchNorm1d(dim_output),
            nn.ReLU()
        )

    def get_output_dim(self):
        return 512

    def forward(self, x):
        batch_size, num_frames, height, width = x.shape
        x_reshaped = x.view(-1, 1, height, width)  # [B*30, 1, 256, 256]
        frame_features = self.feature_extractor(x_reshaped)  # [B*30, 512]
        frame_features = frame_features.view(batch_size, num_frames, -1)
        features = frame_features.view(batch_size, -1)  # [B, 512*30]
        features = self.time_fusion(features)  # [B, 512]
        return features


# === Norm MRI Net
class norm_mri_Net(nn.Module):
    def __init__(self, feature_dim=16):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(
            5, 64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        self.resnet.fc = nn.Identity()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, 256, 256)
        Returns:
            features: (batch_size, feature_dim)
        """
        features = self.resnet(x)  # [batch_size, 512]
        features = self.feature_layer(features)  # [batch_size, feature_dim]
        
        return features


# === MIC Net
class MIC_Net(nn.Module):
    def __init__(self, feature_dim=16):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        self.resnet.fc = nn.Identity()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, 256, 256)
        Returns:
            features: (batch_size, feature_dim)
        """
        features = self.resnet(x)  # [batch_size, 512]
        features = self.feature_layer(features)  # [batch_size, feature_dim]
        
        return features


# === MultiModalDeepSurv
class MultiModalDeepSurv(nn.Module):
    def __init__(self, feature_dim=16, fusion_dim=22, middle_dim=6, ablation=None):
        super(MultiModalDeepSurv, self).__init__()
        
        if ablation == "no_pret":
            self.encoder1 = norm_mri_Net(feature_dim=feature_dim)
            self.encoder2 = norm_mri_Net(feature_dim=feature_dim)
            self.encoder3 = norm_mri_Net(feature_dim=feature_dim)
        else:
            self.encoder1 = MRI_Net(dim_output=feature_dim)
            self.encoder2 = MRI_Net(dim_output=feature_dim)
            self.encoder3 = MRI_Net(dim_output=feature_dim)
        self.encoder4 = MIC_Net(feature_dim=feature_dim)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 4, 128),  # 2048 -> 512
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, middle_dim),      # 512 -> 128
            nn.BatchNorm1d(middle_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.deep_surv = nn.Sequential(
            nn.Linear(fusion_dim+middle_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 2)  # 风险评分，8周内疾病进展风险
        )
        
    def forward(self, input1, input2, input3, input4, input5):
        """
        Args:
            input1-4:
            encoders: [encoder1, encoder2, encoder3, encoder4]
        Returns:
            risk_score:
            features:
        """
        # 提取特征
        feat1 = self.encoder1(input1).squeeze(1)   # [B, 512]
        feat2 = self.encoder2(input2).squeeze(1)   # [B, 512]
        feat3 = self.encoder3(input3).squeeze(1)   # [B, 512]
        feat4 = self.encoder4(input4).squeeze(1)   # [B, 512]
        fused = torch.cat([feat1, feat2, feat3, feat4], dim=1) 
        fused = self.fusion_layer(fused)
        fused_features = torch.cat([input5, fused], dim=1)
        outputs = self.deep_surv(fused_features)
        risk_pred = outputs[:, 0]
        prog_pred = outputs[:, 1]
        return risk_pred, prog_pred



