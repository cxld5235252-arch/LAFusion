# ReWiTe: Realistic Wide-angle and Telephoto Dual Camera Fusion Dataset and Loosely Alignement Based Fusion

## Environment Setup

To set up your environment, follow these steps:

```
conda create -n my_env python=3.8 -y
conda activate my_env
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -y
pip install -r requirements.txt
```

## Visualize Input W and Input T
<p align="center">
  <img src="https://raw.githubusercontent.com/cxld5235252-arch/LAFusion/main/result/284_wide.png" alt="Figure 1" width="45%" style="display:inline-block; margin-right:10px;">
  <img src="https://raw.githubusercontent.com/cxld5235252-arch/LAFusion/main/result/284_wide.png" alt="Figure 2" width="45%" style="display:inline-block;">
</p>

<p align="center"><b>Left Input W, right is Input T.</b></p>



## Test our LAFusion network

Run the following command to test our LAFusion network. Results are saved in the `[result_dir]` folder.
```
CUDA_VISIBLE_DEVICES=0 python inference_LAFusion.py -i ./data/wide_tile -o [result_dir]
```

## Merge tile to full result
```
python /data/cxl/cxl_oppo/github/LGSR/merge_tile.py
```

## Visualize the result 
![Fusion Result](https://github.com/cxld5235252-arch/LAFusion/blob/main/result/284_wide.png)

<p align="center">
  <img src="https://raw.githubusercontent.com/cxld5235252-arch/LAFusion/main/result/284_wide.png" alt="Figure 3" style="display:inline-block; margin-right:10px;">

<p align="center"><b>Figure 1: Left is Input W, right is Input T.</b></p>
