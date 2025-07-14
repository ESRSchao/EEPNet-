import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from models.resnet import Image_ResNet


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    oa = F.logsigmoid(z0)
    ob = F.logsigmoid(z1)
    certainties = oa + ob.transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores, oa, ob

def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1

class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores, oa, ob = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim, oa, ob

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.resnet2d = Image_ResNet()
        self.resnet3d = Image_ResNet()

        self.img_score_head_2d=nn.Sequential(
            nn.Conv2d(64+512,128,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,1,bias=False),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,1,1,bias=False),
            nn.Sigmoid())
        self.img_score_head_3d=nn.Sequential(
            nn.Conv2d(64+512,128,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,1,bias=False),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,1,1,bias=False),
            nn.Sigmoid())

        self.img_feature_layer_2d=nn.Sequential(nn.Conv2d(576,256,1,bias=False),nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(256,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False))
        self.img_feature_layer_3d=nn.Sequential(nn.Conv2d(576,256,1,bias=False),nn.BatchNorm2d(256),nn.ReLU(),nn.Conv2d(256,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False))
        self.log_assignment = MatchAssignment(64)

    def forward(self, image2d, image3d, kp2d, kp3d):
        global_img_feat2d, pixel_wised_feat2d = self.resnet2d(image2d)
        global_img_feat3d, pixel_wised_feat3d = self.resnet3d(image3d)

        img_feat_fusion2d = torch.cat((pixel_wised_feat2d, global_img_feat3d.unsqueeze(-1).unsqueeze(-1).repeat(1,1,pixel_wised_feat2d.shape[2],pixel_wised_feat2d.shape[3])), dim=1)
        img_feat_fusion3d = torch.cat((pixel_wised_feat3d, global_img_feat2d.unsqueeze(-1).unsqueeze(-1).repeat(1,1,pixel_wised_feat3d.shape[2],pixel_wised_feat3d.shape[3])), dim=1)
        
        # img_score2d = self.img_score_head_2d(img_feat_fusion2d)
        # img_score3d = self.img_score_head_3d(img_feat_fusion3d)

        pixel_wised_feat2d = self.img_feature_layer_2d(img_feat_fusion2d)
        pixel_wised_feat3d = self.img_feature_layer_3d(img_feat_fusion3d)
        pixel_wised_feat2d = F.normalize(pixel_wised_feat2d, dim=1,p=2)
        pixel_wised_feat3d = F.normalize(pixel_wised_feat3d, dim=1,p=2)
        
        keypoints_2d = [kp2d[i].squeeze(0) for i in range(kp2d.shape[0])]
        keypoints_3d = [kp3d[i].squeeze(0) for i in range(kp3d.shape[0])]
        des2d = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints_2d, pixel_wised_feat2d)]
        des3d = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints_3d, pixel_wised_feat3d)]
        des2d = torch.stack(des2d)
        des3d = torch.stack(des3d)
        match_list = []
        sim_list = []
        scores0 = []
        scores1 = []

        for i in range(des3d.shape[0]):
            scores, sim, oa, ob = self.log_assignment(des3d[i].transpose(1, 0).unsqueeze(0), des2d[i].transpose(1, 0).unsqueeze(0))  
            scores0.append(oa)
            scores1.append(ob)
            
            m0, m1, mscores0, mscores1 = filter_matches(scores, 0.0)
            valid = m0[0] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[0][valid]
            match_list.append(torch.stack([m_indices_0, m_indices_1], -1))
            sim_list.append(scores.squeeze(0))  

        return match_list, sim_list, scores0, scores1
