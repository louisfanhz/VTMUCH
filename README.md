# VTMUCH

This is the repo for the 2025 AAAI conference paper "Vision-guided Text Mining for Unsupervised Cross-modal Hashing with Community Similarity Quantization". 

The following is the pipeline for VTMUCH:



## Installation

1. Create and activate a virtual environment
2. Install dependencies:
```bash
pip install torch==2.4.1 torchvision==0.19.1 ftfy mmdet openmim cdlib leidenalg
mim install mmcv==2.1.0
```

## Setup

1. Download and unzip the Flickr25K dataset to `./VTMUCH/dataset/flickr25k`
2. Run the dataset preparation script:
```bash
cd ./VTMUCH/dataset/flickr25k
python make_mirflickr25k.py
```
3. Download CLIP weights (ViT-B-32.pt) and place them in the `./cache` directory

## Usage

### Training

To train the model:
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
