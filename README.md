# ReWiTe: Realistic Wide-angle and Telephoto Dual Camera Fusion Dataset and Loosely Alignement Based Fusion

## Environment Setup

To set up your environment, follow these steps:

```
conda create -n my_env python=3.8 -y
conda activate my_env
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -y
pip install -r requirements.txt
```

## Test our LAFusion network

Run the following command to test our LAFusion network. Results are saved in the `[result_dir]` folder.
```
python inference_LAF.py -i [wide_dir] -o [result_dir]
```

## data
Prepare data.
```
gt_root: LAFusion/data_new_gt_0925/train/gt_y_128
lq_root: LAFusion/data_new_gt_0925/train/wide_y_128
tele_root: LAFusion/data_new_gt_0925/train/ref_y_128
```

