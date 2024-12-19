# VTMUCH

use a virtual environment

please intall dependencies as: <br />
pip install torch==2.4.1 torchvision==0.19.1 tensorrt ftfy mmdet openmim cdlib leidenalg
mim install mmcv==2.1.0

please unzip the flickr25k images to "./VTMUCH/dataset/flickr25k"
then run "python make_mirflickr25k.py" within flickr25k directory

to reproduce experiment use:
python main.py --is-train --valid-freq=1 --epochs=20 --n-clusters=14 --dataset=flickr25k --k-bits=32 --batch-size=128 --detection-score=0.3 --si-std=1.1

to test model and save hash codes please use:
python main.py --pretrained=model_path --k-bits=16 --dataset=flickr25k
that will save a dictionary containing binary codes to "./result"

PR is obtained by:
draw_range = np.linspace(1, 18015, 181, dtype=int)
precision_16_i2t, recall_16_i2t = pr_curve(result_16["r_txt"], result_16["q_img"], result_16["r_l"], result_16["q_l"], draw_range)
precision_16_t2i, recall_16_t2i = pr_curve(result_16["r_img"], result_16["q_txt"], result_16["r_l"], result_16["q_l"], draw_range)
