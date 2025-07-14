import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def replace_nan_values(point_set):
    """
    将点集中的NaN值替换为(0, 0)
    Args:
        point_set (torch.Tensor): 输入的点集，形状为(batch_size, 2, num_points)
    Returns:
        cleaned_point_set (torch.Tensor): 替换NaN值后的点集，形状为(batch_size, 2, num_points)
    """
    point_set.permute(0, 2, 1)
    # 检测NaN值
    nan_mask = torch.isnan(point_set)
    
    # 替换NaN值为(0, 0)
    cleaned_point_set = torch.where(nan_mask, torch.tensor([0.0, 0.0], device=point_set.device), point_set)
    cleaned_point_set.permute(0, 2, 1)
    return cleaned_point_set


def proj(xy_points, batch_size, H, W):
    image_mask = torch.zeros(batch_size, W, H, 1)  # 初始化图像，假设所有像素值为 0
    for point in range(xy_points.shape[1]):
        # 获取点的投影位置
        x = xy_points[0][point][0]
        y = xy_points[0][point][1]
    
        # 将点投影到图像上
        if 0 <= x < W and 0 <= y < H:
            image_mask[0, int(x), int(y), 0] = 1
    return image_mask
    # plt.title("Velodyne points to camera image Result")
    # plt.imshow(image_mask[0])
    # plt.show()

def tr3d2d(pointclouds, inter_matrix, transform_matrix, T, H, W):
    pointclouds = torch.cat(
                (pointclouds, torch.ones(pointclouds.size(0), pointclouds.size(1), 1).to(pointclouds.device)),
                dim=2)
    inter_matrix = inter_matrix[:, :, :3]  # 现在 P0 的形状为 (batch_size, 3, 3)
    transform_matrix = torch.bmm(transform_matrix.float(), T.float())
    points = torch.matmul(pointclouds.to('cpu').float(), transform_matrix.transpose(1, 2))
    points = torch.matmul(points, inter_matrix.transpose(1, 2))
    z_coords = points[:, :, 2:3]
    
    positive_mask = (points >= 0).all(dim=2)
    positive_mask_z = (z_coords >= 0).all(dim=2)    
    # 将所有不满足条件的坐标置零
    points[~positive_mask] = 0
    z_coords[~positive_mask_z] = 0
    points = points / z_coords
    xy_points = points[:, :, :2]

    xy_points = replace_nan_values(xy_points)
    fov_mask = (xy_points[:, :, 0] <= W) & (xy_points[:, :, 1] <= H)
    xy_points[~fov_mask] = 0
    heatmap = proj(xy_points, pointclouds.size(0), H, W)
    heatmap = heatmap.permute(0, 3, 2, 1)

    return heatmap, xy_points