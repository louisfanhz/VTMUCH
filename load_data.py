from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision
import torchvision.transforms as T 
from model.simple_tokenizer import SimpleTokenizer
import numpy as np
import scipy.io as scio


class BaseDataset(Dataset):
    def __init__(self,
                 captions: dict,
                 BoWs: dict,
                 indexs: dict,
                 labels: dict,
                 is_train=True,
                 tokenizer=SimpleTokenizer(),
                 maxWords=32,
                 imageResolution=224,
                 ):
        
        self.captions = [','.join(s.strip().split()) for s in captions]
        self.BoWs = BoWs
        self.indexs = indexs
        self.labels = labels
        self.maxWords = maxWords
        self.tokenizer = tokenizer

        self.transform = T.Compose([
            T.Resize(imageResolution, interpolation=Image.BICUBIC),
            T.CenterCrop(imageResolution),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) if is_train else T.Compose([
            T.Resize((imageResolution, imageResolution), interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        
        self.__length = len(self.indexs)

    def __len__(self):
        return self.__length
    
    def _load_image(self, index: int) -> torch.Tensor:
        image_path = self.indexs[index].strip()
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image
    
    def _load_text(self, index: int):
        captions = self.captions[index]
        words = self.tokenizer.tokenize(captions)
        words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
        total_length_with_CLS = self.maxWords - 1
        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]
        words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
        caption = self.tokenizer.convert_tokens_to_ids(words)
        while len(caption) < self.maxWords:
            caption.append(0)
        caption = torch.tensor(caption)
        key_padding_mask = (caption == 0)
        return caption, key_padding_mask
    
    def get_all_label(self):
        labels = torch.zeros([self.__length, len(self.labels[0])], dtype=torch.int64)
        for i, item in enumerate(self.labels):
            labels[i] = torch.from_numpy(item)
        return labels
    
    def get_all_image_path(self):
        paths = []
        for path in self.indexs:
            paths.append(path.strip())
        return paths
    
    def __getitem__(self, index):
        image = self._load_image(index)
        caption, key_padding_mask = self._load_text(index)

        return image, caption, key_padding_mask, index
    

class DetectionDataset(Dataset):
    def __init__(self, indexs: dict):
        self.indexs = indexs
        self.__length = len(self.indexs)
        self.transform = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
    
    def __len__(self):
        return self.__length
    
    def _load_image(self, index: int) -> torch.Tensor:
        image_path = self.indexs[index].strip()
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image
    
    def __getitem__(self, index):
        image = self._load_image(index)
        return image, index

def split_data(captions, BoWs, indexs, labels, query_num, train_num, seed=None):
    if seed != None:
        np.random.seed(seed=seed)  # fixed to 1 for all experiments.

    random_index = np.random.permutation(range(len(indexs)))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:] 

    query_indexs = indexs[query_index]
    query_captions = captions[query_index]
    query_labels = labels[query_index]
    query_BoWs = BoWs[query_index]

    train_indexs = indexs[train_index]
    train_captions = captions[train_index]
    train_labels = labels[train_index]
    train_BoWs = BoWs[train_index]

    retrieval_indexs = indexs[retrieval_index]
    retrieval_captions = captions[retrieval_index]
    retrieval_labels = labels[retrieval_index]
    retrieval_BoWs = BoWs[retrieval_index]

    split_indexs = (query_indexs, train_indexs, retrieval_indexs)
    split_captions = (query_captions, train_captions, retrieval_captions)
    split_labels = (query_labels, train_labels, retrieval_labels)
    split_BoWs = (query_BoWs, train_BoWs, retrieval_BoWs)

    return split_indexs, split_captions, split_labels, split_BoWs

def generate_dataset(captionFile: str,
                    bowFile: str,
                    indexFile: str,
                    labelFile: str,
                    maxWords=32,
                    imageResolution=224,
                    query_num=2000,
                    train_num=5000,
                    seed=None,
                    ):
    
    captions = scio.loadmat(captionFile)["caption"]
    BoWs = scio.loadmat(bowFile)["caption_one_hot"]
    indexs = scio.loadmat(indexFile)["index"]
    labels = scio.loadmat(labelFile)["label"]

    split_indexs, split_captions, split_labels, split_BoWs = split_data(captions, BoWs, indexs, labels, query_num=query_num, train_num=train_num, seed=seed)
        
    query_data = BaseDataset(captions=split_captions[0], BoWs=split_BoWs[0], indexs=split_indexs[0], labels=split_labels[0],
                            maxWords=maxWords, imageResolution=imageResolution, is_train=False)
    train_data = BaseDataset(captions=split_captions[1], BoWs=split_BoWs[1], indexs=split_indexs[1], labels=split_labels[1],
                            maxWords=maxWords, imageResolution=imageResolution)
    retrieval_data = BaseDataset(captions=split_captions[2], BoWs=split_BoWs[2], indexs=split_indexs[2], labels=split_labels[2],
                                maxWords=maxWords, imageResolution=imageResolution, is_train=False)

    return train_data, query_data, retrieval_data