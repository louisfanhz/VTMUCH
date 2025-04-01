# VTMUCH

This is the repo for the 2025 AAAI conference paper "Vision-guided Text Mining for Unsupervised Cross-modal Hashing with Community Similarity Quantization". 

The following is the pipeline of VTMUCH:
![alt text](https://github.com/louisfanhz/VTMUCH/blob/main/figs/VTMUCH_framework.jpg?raw=true)

Experimental result of VTMUCH:
![alt text](https://github.com/louisfanhz/VTMUCH/blob/main/figs/VTMUCH_exp_results.jpg?raw=true)


## Installation

1. Create and activate a [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) virtual environment from environment.yaml
```bash
conda env create --file=environment.yaml
```
2. Install mmdet following https://github.com/open-mmlab/mmdetection.

## Setup

1. The datasets used in VTMUCH follow official releases of FLICKR25K and NUWIDE. You can also download the datsets manually with the following links:
   
    FLICKR25K: https://drive.google.com/drive/folders/16_n_afpLz2q9R_BTL_FAJaZF5sjTnI5a?usp=sharing
   
    NUSWIDE: https://drive.google.com/drive/folders/1aBCP-38ZyDfgQKI2_nYmITZEFAFY0yXP?usp=sharing

2. Unzip the downloaded image files to `./dataset/flickr25k` and `./dataset/nuswide`, and run the data preparation scripts
```bash
cd ./dataset/flickr25k/
python make_mirflickr25k.py

cd ./dataset/nuswide/
python make_nuswide.py
```
3. Download CLIP weights (ViT-B-32.pt) from the following link and place them in the `./cache` directory:
https://drive.google.com/file/d/1KRCPSx0-wZLUIkb1ru1laLioexL2Efuk/view?usp=sharing

## Usage

### Training

Below is an example command to run the training pipeline of VTMUCH. Here we are training with 0.3 threshold score for object detection, 14 KMeans clusters for detected objects, and 1.1 standard deviation cutoff for S_I matrix construction. For more detail, please refer to our original paper.
```bash
python main.py --is-train \
    --valid-freq=1 \
    --epochs=20 \
    --n-clusters=14 \
    --dataset=flickr25k \
    --k-bits=16 \
    --batch-size=128 \
    --detection-score=0.3 \
    --si-std=1.1
```

### Testing

To test the model and generate hash codes:
```bash
python main.py --pretrained=model_path --k-bits=16 --dataset=flickr25k
```
This will save the binary codes to the `./result` directory.

### Evaluation

To calculate Precision-Recall curves:
```python
draw_range = np.linspace(1, 18015, 181, dtype=int)
precision_16_i2t, recall_16_i2t = pr_curve(result_16["r_txt"], result_16["q_img"], result_16["r_l"], result_16["q_l"], draw_range)
precision_16_t2i, recall_16_t2i = pr_curve(result_16["r_img"], result_16["q_txt"], result_16["r_l"], result_16["q_l"], draw_range)
```
