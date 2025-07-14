import os
import math
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from scipy.spatial.transform import Rotation
import cv2

class KITTIDataset(Dataset):
    def __init__(self, data_root, sequence_range, proj_H, proj_W, trans_R, trans_T, mode,shuffle_data=False):
        self.data_root = data_root
        self.sequence_range = sequence_range
        self.data = self.load_data()
        if shuffle_data:
            random.shuffle(self.data)  

    def load_data(self):
        data = []
        for sequence_num in self.sequence_range:
            sequence_dir = os.path.join(self.data_root, f"sequences/{sequence_num:02d}")
            file2_dir = os.path.join(sequence_dir, "test_P2_ACC")
            file3_dir = os.path.join(sequence_dir, "test_P3_ACC")

            file2_name = sorted(os.listdir(file2_dir))
            file3_name = sorted(os.listdir(file3_dir))

            for file2, file3 in zip(file2_name, file3_name):
                file2_path = os.path.join(file2_dir, file2)
                file3_path = os.path.join(file3_dir, file3)
                data.append(file2_path)
                data.append(file3_path)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data[idx]

        with open(filename, 'rb') as fin:
            pc3d = np.frombuffer(fin.read(3000 * 3 * 4), dtype=np.float32).reshape(-1, 3)
            image_2d = np.frombuffer(fin.read(160 * 512 * 2), dtype=np.int16).reshape(160, 512)
            image_3d = np.frombuffer(fin.read(64 * 1024 * 2), dtype=np.int16).reshape(64, 1024)
            kp2d = np.frombuffer(fin.read(3000 * 2 * 2), dtype=np.int16).reshape(-1, 2)
            kp3d = np.frombuffer(fin.read(3000 * 2 * 2), dtype=np.int16).reshape(-1, 2)
            inter_matrix = np.frombuffer(fin.read(3 * 4 * 4), dtype=np.float32).reshape(-1, 4)
            transform_matrix = np.frombuffer(fin.read(3 * 4 * 4), dtype=np.float32).reshape(-1, 4)
            T_inv = np.frombuffer(fin.read(4 * 4 * 4), dtype=np.float32).reshape(-1, 4)


            pc_3d = torch.from_numpy(pc3d.astype(np.float32))
            image_2d = torch.from_numpy(image_2d.astype(np.float32) / 255.)
            image_3d = torch.from_numpy(image_3d.astype(np.float32) / 255.)
            kp2d = torch.from_numpy(kp2d.astype(np.float32))
            kp3d = torch.from_numpy(kp3d.astype(np.float32))

        return pc_3d ,image_2d, image_3d, kp2d, kp3d, inter_matrix, transform_matrix, T_inv
