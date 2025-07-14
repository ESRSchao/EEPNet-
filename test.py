import torch
import torch.nn
from KITTI_bin import KITTIDataset
from models import network, vis
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import argparse
from PIL import Image, ImageDraw
import numpy as np
import torch.nn.functional as F
import cv2
from scipy.spatial.transform import Rotation
import pandas as pd
import time
seed_value = 42 

torch.manual_seed(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./KTO')
parser.add_argument("--device", type=str, default='cuda:1')
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--trans_T", type=float, default=10)
parser.add_argument("--trans_R", type=float, default=2 * np.pi)
parser.add_argument("--proj_H", type=float, default=64)
parser.add_argument("--proj_W", type=float, default=1024)

args = parser.parse_args()

dataset = args.dataset_path
batch_size = args.batch_size
trans_T = args.trans_T
trans_R = args.trans_R
proj_H = args.proj_H
proj_W = args.proj_W

if torch.cuda.is_available():
    device = torch.device(args.device)
    print(f"device: {device}")
else:
    device = torch.device("cpu")
    print("CUDA not avialable，use CPU。")

if __name__=='__main__':
    data = dict()
    data['RRE'] = []
    data['RTE'] = []
    data_root = dataset
    sequence_range_test = [9, 10]  
    test_data = KITTIDataset(data_root, sequence_range_test, proj_H, proj_W, trans_R=trans_R, trans_T=trans_T, mode='test')
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)
    num_train = len(test_data)
    print('Train dataset size: ', num_train)

    inst_network = network.MainNet().to(device)

    loaded_checkpoint = torch.load('./checkpoints/R/Encoder_epoch_30.t7', map_location='cpu')
    inst_network.load_state_dict(loaded_checkpoint )
    inst_network.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch in enumerate(test_loader):
            pc_3d ,image_2d, image_3d, kp2d, kp3d, inter_matrix, transform_matrix, T = batch
            pc_3d = pc_3d.to(device)
            kp2d = kp2d.to(device)
            kp3d = kp3d.to(device)
            image2d = image_2d.unsqueeze(1).to(device)
            image3d = image_3d.unsqueeze(1).to(device)
            inter_matrix = inter_matrix.float()
            transform_matrix = transform_matrix.float()
            T_real = torch.bmm(transform_matrix, T)

            match_list, sim_list, score0, score1 = inst_network(image2d, image3d, kp2d, kp3d)

            H, W = image_2d.size(1), image_2d.size(2)
            heatmap, xy_points = vis.tr3d2d(pc_3d, inter_matrix, transform_matrix, T, H, W)

            for b in range(batch_size):

                GT_feature_points = kp2d[b].to('cpu').numpy()  
                pre_feature_points = xy_points[b].to('cpu').numpy()
                mat = match_list[b].to('cpu')

                x_coords1 = np.array(GT_feature_points[mat[:, 1], 0])
                y_coords1 = np.array(GT_feature_points[mat[:, 1], 1])
                x_coords2 = np.array(pre_feature_points[mat[:, 0], 0])
                y_coords2 = np.array(pre_feature_points[mat[:, 0], 1])


                camera_matrix = inter_matrix[b, :, :3].numpy()
                image_points = kp2d[b][mat[:, 1], :].to('cpu').numpy()
                image_points = np.array(image_points, dtype=np.float32)
                world_points = pc_3d[b, mat[:, 0], :].to('cpu').numpy()

                try:
                    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(world_points, image_points,
                                                                                           camera_matrix, None, reprojectionError=6.5, flags=cv2.SOLVEPNP_EPNP)
                
                except: success = False
                if success:
                    # rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    # rotation_matrix = np.concatenate((rotation_matrix, translation_vector), axis=1)   
                    extra_row = np.array([0, 0, 0, 1])
                    GT_Tr = T_real[b].numpy()
                    GT_Tr = np.vstack([GT_Tr, extra_row])
                    GT_R = GT_Tr[:3, :3]
                    GT_R = Rotation.from_matrix(GT_R)
                    GT_RE = GT_R.as_euler('xzy', degrees=True)
                    GT_T = GT_Tr[:3, 3]
                    R,_=cv2.Rodrigues(rotation_vector)
                    T_pred=np.eye(4)
                    T_pred[0:3,0:3] = R
                    T_pred[0:3,3:] = translation_vector
                    P_diff=np.dot(np.linalg.inv(T_pred),GT_Tr)
                    t_diff=np.linalg.norm(P_diff[0:3,3])
                    r_diff=P_diff[0:3,0:3]
                    R_diff=Rotation.from_matrix(r_diff)
                    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
                    rte = t_diff
                    rre = angles_diff
                    print(rre, rte)
                    print('success')
                else:
                    rre = np.inf
                    rte = np.inf
                    print('G')
                data['RRE'].append(rre)
                data['RTE'].append(rte)

        df = pd.DataFrame(data)

        excel_file_path = 'R30.xlsx'
        
        df.to_excel(excel_file_path, index=True)
        end_time = time.time()
        run_time = end_time - start_time
        print("Run Time：", run_time, "s")