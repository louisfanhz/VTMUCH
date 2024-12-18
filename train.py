import os
import  torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json

from optimization import BertAdam
from load_data import generate_dataset
from utils import get_logger
from utils.calc_utils import calc_mAP_k
from model.vtmuch import VTMUCH

dataset_root_path = "./dataset"

class Train:
    def __init__(self, args):

        self.args = args
        self.args.dataset_root_path = dataset_root_path

        np.random.seed(self.args.seed)
        torch.random.manual_seed(seed=self.args.seed)
        torch.autograd.set_detect_anomaly(True)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.makedirs(self.args.save_dir, exist_ok=True)
        self._init_writer()

        self.logger.info('Start logging...')

        self.rank = self.args.rank  # gpu rank

        self._init_dataset()
        self._init_model()

        self.run_stat = {'best_i2t': 0, 'best_t2i': 0, 'best_i2i': 0, 'best_t2t': 0}
        
        self.logger.info("done initializing")
        self.run()

    def _init_writer(self):
        self.logger = get_logger(os.path.join(self.args.save_dir, "train.log" if self.args.is_train else "test.log"))
        with open(os.path.join(self.args.save_dir, 'all_args.txt'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

    def _init_dataset(self):
        self.logger.info("initializing dataset...")
        self.logger.info(f"Using {self.args.dataset} dataset...")
        self.logger.info(f"Detection dir: {self.args.detection_dir}")

        global dataset_root_path
        self.args.index_file = os.path.join(dataset_root_path, self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join(dataset_root_path, self.args.dataset, self.args.caption_file)
        self.args.bow_file = os.path.join(dataset_root_path, self.args.dataset, self.args.bow_file)
        self.args.label_file = os.path.join(dataset_root_path, self.args.dataset, self.args.label_file)

        train_data, query_data, retrieval_data = generate_dataset(captionFile=self.args.caption_file,
                                                                bowFile=self.args.bow_file,
                                                                indexFile=self.args.index_file,
                                                                labelFile=self.args.label_file,
                                                                maxWords=self.args.max_words,
                                                                imageResolution=self.args.resolution,
                                                                query_num=self.args.query_num,
                                                                train_num=self.args.train_num,
                                                                seed=self.args.seed)
        
        self.train_labels = train_data.get_all_label().float()
        self.query_labels = query_data.get_all_label().float()
        self.retrieval_labels = retrieval_data.get_all_label().float()

        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"train labels shape: {self.train_labels.shape}")
        self.logger.info(f"query labels shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval labels shape: {self.retrieval_labels.shape}")

        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=False,
            shuffle=True
        )

    def _init_model(self):
        self.logger.info("initializing model...")

        self.model = VTMUCH(self.args, self.train_loader).to(self.rank).float()

        self.optimizer = BertAdam(
            [
                {'params': self.model._clip.parameters(), 'lr': self.args.clip_lr},
                {'params': self.model.imgNet.parameters(), 'lr': self.args.lr},
                {'params': self.model.txtNet.parameters(), 'lr': self.args.lr},
            ],
            lr=self.args.lr,
            warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)        

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.args.save_dir, "model.pth"))
        self.logger.info(f"{self.args.dataset}_{self.args.k_bits}. Epoch: {epoch}, save model to {os.path.join(self.args.save_dir, 'model.pth')}")

    def run(self):
        if self.args.is_train:
            self.train()
        else:
            self.test()

    def test(self):
        if self.args.pretrained == "" or self.args.pretrained == "MODEL_PATH":
            self.logger.error("test step must load a model! please set the --pretrained argument.")
            raise RuntimeError("test step must load a model! please set the --pretrained argument.")

        self.model.set_eval()

        q_i, q_t = self.get_code(self.query_loader, self.args.query_num)
        r_i, r_t = self.get_code(self.retrieval_loader, self.args.retrieval_num)

        _k_ = None
        mAPi2t = calc_mAP_k(q_i.to(self.device), r_t.to(self.device), self.query_labels.to(self.device),
                            self.retrieval_labels.to(self.device), _k_).item()
        mAPt2i = calc_mAP_k(q_t.to(self.device), r_i.to(self.device), self.query_labels.to(self.device),
                            self.retrieval_labels.to(self.device), _k_).item()

        self.logger.info(f"MAP(i->t): {round(mAPi2t, 5)}, MAP(t->i): {round(mAPt2i, 5)}")
        self.logger.info(">>>>>> Save *.mat data! Exit...")

    def valid(self, epoch):
        self.logger.info("\n")
        self.logger.info(" Validating: %d/%d " % (epoch, self.args.epochs))
        self.model.set_eval()

        with torch.no_grad():
            q_i, q_t = self.get_code(self.query_loader, self.args.query_num)
            r_i, r_t = self.get_code(self.retrieval_loader, self.args.retrieval_num)

            _k_ = None
            mAPi2t = calc_mAP_k(q_i.to(self.rank), r_t.to(self.rank), self.query_labels.to(self.rank),
                                self.retrieval_labels.to(self.rank), _k_).item()
            mAPt2i = calc_mAP_k(q_t.to(self.rank), r_i.to(self.rank), self.query_labels.to(self.rank),
                                self.retrieval_labels.to(self.rank), _k_).item()
            
            # for comparing with other baslines: mAP50
            mAPi2t50 = calc_mAP_k(q_i.to(self.rank), r_t.to(self.rank), self.query_labels.to(self.rank),
                                self.retrieval_labels.to(self.rank), 50).item()
            mAPt2i50 = calc_mAP_k(q_t.to(self.rank), r_i.to(self.rank), self.query_labels.to(self.rank),
                                self.retrieval_labels.to(self.rank), 50).item()

            if mAPi2t + mAPt2i > self.run_stat['best_i2t'] + self.run_stat['best_t2i']:
                self.run_stat['best_i2t'] = mAPi2t
                self.run_stat['best_t2i'] = mAPt2i
                self.logger.info("$$$$$$$$$$$$$$$$$$$$ New Best $$$$$$$$$$$$$$$$$$$$$$$$")
                self.save_model(epoch)

            self.logger.info(f"mAPi2t: {mAPi2t}, mAPt2i: {mAPt2i}")
            self.logger.info(f"mAPi2t50: {mAPi2t50}, mAPt2i50: {mAPt2i50}")

    def train(self):
        self.logger.info("Start training...")

        for epoch in range(self.args.epochs):
            self.logger.info(f"Training epoch {epoch}: ")

            self.train_one_epoch(epoch)

            if epoch == 0 or (epoch + 1) % self.args.valid_freq == 0:
                self.valid(epoch)

    def train_one_epoch(self, epoch):
        self.model.set_train()

        for image, caption, key_padding_mask, index in tqdm(self.train_loader, 
                                                            desc=f"epoch {epoch} progress"):
            image = image.float().to(self.rank, non_blocking=True)
            caption = caption.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)

            output_dict = self.model(index, image, key_padding_mask, caption)

            ALL_LOSS = self.compute_loss(output_dict)

            loss = 0
            for key in ALL_LOSS:
                loss += ALL_LOSS[key]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_loss(self, output_dict):
        ALL_LOSS = {}

        txt_hash = output_dict['txt_hash']
        img_hash = output_dict['img_hash']
        SF = output_dict['SF']

        BI_BI = img_hash.mm(img_hash.T)
        BT_BT = txt_hash.mm(txt_hash.T)
        BI_BT = img_hash.mm(txt_hash.T)

        sim_preservation_loss = 0.1*F.mse_loss(BT_BT, self.args.k_bits * SF) + \
                                0.9*F.mse_loss(BI_BI, self.args.k_bits * SF) + \
                                F.mse_loss(BI_BT, self.args.k_bits * SF)
        ALL_LOSS['sim_preservation_loss'] = sim_preservation_loss

        semantic_alignment_loss = F.mse_loss(BI_BI, BT_BT) + \
                                  F.mse_loss(BI_BT, BI_BI) + \
                                  F.mse_loss(BI_BT, BT_BT)
        ALL_LOSS['semantic_alignment_loss'] = semantic_alignment_loss

        return ALL_LOSS
    
    def get_code(self, data_loader, length: int):
        k_bits = self.args.k_bits

        img_buffer = torch.empty(length, k_bits, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(length, k_bits, dtype=torch.float).to(self.rank)

        for image, caption, key_padding_mask, index in tqdm(data_loader,
                                                            desc=f"Validating: loading batch"):
            image = image.float().to(self.rank, non_blocking=True)
            caption = caption.to(self.rank, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.rank, non_blocking=True)
            
            output_dict = self.model(index, image, key_padding_mask, caption)

            txt_hash = output_dict['txt_hash'].detach()
            img_hash = output_dict['img_hash'].detach()

            img_buffer[index, :] = torch.sign(img_hash)
            text_buffer[index, :] = torch.sign(txt_hash)

        return img_buffer, text_buffer