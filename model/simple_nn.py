import torch
import torch.nn as nn
import torch.nn.functional as F

class ImgNet(nn.Module):
    def __init__(self, k_bits, img_feat_len):
        super(ImgNet, self).__init__()
        self.fc1 = nn.Linear(img_feat_len, 2048)
        self.fc2 = nn.Linear(2048, k_bits)

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = F.tanh(hid)
        return code

class TxtNet(nn.Module):
    def __init__(self, k_bits, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 2048)
        self.fc2 = nn.Linear(2048, k_bits)

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = F.tanh(hid)
        return code
    

