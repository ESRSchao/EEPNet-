# EEPNet: Efficient Edge Point-based Matching Network for Cross-Modal Dynamic Registration between LiDAR and Camera

This repository is the official implementation of [EEPNet: Efficient Edge Point-based Matching
Network for Cross-Modal Dynamic Registration
between LiDAR and Camera]
![EEPNet Best Paper Award](./logo.png)
## Environment
You can set up the Python environment using the following command:
``` python
conda create -n regis2D_y3D python==3.8.5 -y

conda activate regis2D_3D

pip install numpy pillow opencv-python scipy pandas matplotlib
```

---

If need GPU for training and testing, install the appropriate [PyTorch](https://pytorch.org/) version for your GPU drivers:

```python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Training

If you need to perform training, please download the KITTI Odometry dataset from the KITTI official website and fill in the path to the dataset in the command below:

```train
python train.py --dataset_path <path_to_dataset> --device cuda:0
```

## Evaluation

We provide a packaged test dataset for reproducing the results presented in the paper. You can download this dataset using this [link](https://drive.google.com/file/d/1GOwlK_K29-63kfp_0RZ0-w7o4hGDuRYb/view?usp=sharing). Additionally, you need to download our pre-trained models, which can be found at the [link](https://drive.google.com/file/d/1bdZhXDj4foHyO3gCcBJv12BesEudj395/view?usp=sharing). After downloading, place the dataset in the root directory and start the evaluation with the following command:

```eval
python test.py
```

After completing the evaluation process, an xlsx file summarizing all the results will be generated.

## Results

Our model achieves the following performance on :

| Method                          | εr | RTE(m)   | RRE(°)   | Acc.  | Total Time(s) | Pose Inference(s) |
|---------------------------------|----|----------|----------|-------|---------------|-------------------|
| Reflectance Map                 | 8.0| 0.55±0.59| 3.01±3.06| 85.67 | 76.021        | 0.0136            |
|                                 | 6.0| 0.54±0.85| 2.97±3.05| 85.74 | 84.802        | 0.0152            |
|                                 | 4.0| 0.54±0.78| 2.99±3.04| 85.92 | 97.199        | 0.0174            |
|                                 | 2.0| 0.57±0.45| 3.08±1.97| 83.75 | 98.931        | 0.0177            |
| Depth Map                       | 8.0| 0.85±2.55| 3.45±7.86| 80.93 | 81.698        | 0.0146            |
|                                 | 6.0| 0.81±2.19| 3.43±8.34| 81.51 | 90.454        | 0.0162            |
|                                 | 4.0| 0.77±1.66| 3.33±6.00| 80.48 | 98.842        | 0.0177            |
|                                 | 2.0| 0.80±0.66| 3.36±2.27| 78.03 | 98.434        | 0.0176            |



## Contributing

MIT License

Copyright (c) [year] [fullname]
