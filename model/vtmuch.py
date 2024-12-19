import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as T
import numpy as np
import pickle
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans
from mmdet.apis import DetInferencer

from model.simple_nn import ImgNet, TxtNet
# from model.resnet import resnet34, resnet50
from utils.calc_utils import calc_pairwise_cos_sim
from model.clip import load_download_clip
import networkx as nx
import cdlib
from cdlib.algorithms import leiden
from scipy.linalg import hadamard


class ObjectDetection:
    def __init__(self, args, model_infer, train_loader=None):
        self.args = args

        if self.args.detection_dir == "":
            self.args.detection_dir = "images_detected"
            self.args.detection_dir = os.path.join(self.args.dataset_root_path, self.args.dataset, self.args.detection_dir)
            if not os.path.exists(self.args.detection_dir):
                os.makedirs(self.args.detection_dir)

            self.inferencer1 = DetInferencer('faster-rcnn_r50_fpn_32x2_cas_1x_openimages_challenge', 
                                            device=f'cuda:{self.args.rank}', show_progress=False)
            
            self.model_infer = model_infer
            assert not self.model_infer.training

            self.obj_transform = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
            self.clip_transform = T.Compose([
                T.Resize(args.resolution, interpolation=Image.BICUBIC),
                T.CenterCrop(args.resolution),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

            with torch.no_grad():
                self.obj_clip_features = self._perform_detection(train_loader.dataset.get_all_image_path())
        else:
            with open(os.path.join(self.args.detection_dir, 'detection_result.pkl'), 'rb') as f:
                self.obj_clip_features = pickle.load(f)

    def get_obj_clip_features(self):
        return self.obj_clip_features
        
    def _encode_image(self, images):
        images = self.clip_transform(images).unsqueeze(0).to(self.args.rank)

        if getattr(self.model_infer, "encode_image", None):
            # using CLIP
            embeds = self.model_infer.encode_image(images).squeeze()
        else:
            # using ResNet
            embeds = self.model_infer(images).squeeze()

        return embeds
    
    def _prepare_train_dataset(self, img_paths):
        save_dir = os.path.join(self.args.dataset_root_path, self.args.dataset, "images_train")
        if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for i, path in enumerate(img_paths):
            pil_img = Image.open(path)
            pil_img.save(f"{save_dir}/img{i}.jpg")

        return save_dir

    def _perform_detection(self, img_paths):
        obj_clip_features = {}
        for i, path in enumerate(tqdm(img_paths, desc="object detection:")):
            pil_img = Image.open(path)
            np_img = np.asarray(pil_img)
            scores, bboxes = self._detect(np_img)

            pil_img = Image.open(img_paths[i])
            img = F.pil_to_tensor(pil_img)
            # save visualization to directory
            result_img = F.to_pil_image(self._visualize(img, bboxes, scores))
            result_img.save(f"{self.args.detection_dir}/img{i}.jpg")

            # get detected objects as cropped-out images
            objs_cropped = []
            for box in bboxes:
                xmin, ymin, xmax, ymax = box.round().astype(int)
                if xmin == xmax or ymin == ymax:
                    continue
                objs_cropped.append(F.crop(img, ymin, xmin, ymax-ymin, xmax-xmin))

            # encode and save detected objects
            obj_clip_features[i] = []
            for obj_num, obj in enumerate(objs_cropped):
                obj_img = F.to_pil_image(obj)
                obj_img_embed = self._encode_image(obj_img)
                obj_clip_features[i].append(obj_img_embed)
                obj_img.save(f"{self.args.detection_dir}/img{i}_obj{obj_num}.jpg")
            img_embed = self._encode_image(F.to_pil_image(img))
            obj_clip_features[i].append(img_embed)
            obj_clip_features[i] = torch.stack(obj_clip_features[i])

        with open(os.path.join(self.args.detection_dir, 'detection_result.pkl'), 'wb') as f:
            pickle.dump(obj_clip_features, f)

        return obj_clip_features
    
    def _detect(self, img):
        result_model1 = self.inferencer1(img, no_save_vis=True, 
                                   pred_score_thr=self.args.detection_score)['predictions'][0]
        scores = np.array(result_model1['scores'])
        bboxes = np.array(result_model1['bboxes'])

        # only keep boxes above certain confidence score
        score_mask = scores > self.args.detection_score
        scores = scores[score_mask]
        bboxes = bboxes[score_mask]

        # if two boxes are nearly identical, keep only one
        def prune_bboxes_1(scores, bboxes):
            if bboxes.size == 0:
                return scores, bboxes
            similar_bboxes_mask = np.abs(bboxes[:, np.newaxis, :] - bboxes[np.newaxis, :, :])
            similar_bboxes_mask = (similar_bboxes_mask < self.args.bboxes_prune_pixel).all(axis=-1)
            _, discard_indices = np.nonzero(np.triu(similar_bboxes_mask, 1))
            discard_indices = np.unique(discard_indices)
            discard_mask = np.ones((len(bboxes)))
            discard_mask[discard_indices] = 0
            discard_mask = discard_mask.astype(bool)

            return scores[discard_mask], bboxes[discard_mask]

        # remove very small boxes
        def prune_bboxes_2(scores, bboxes):
            if bboxes.size == 0:
                return scores, bboxes
            img_area = img.shape[0]*img.shape[1]
            bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            discard_mask = np.ones((len(bboxes)))
            discard_mask[bboxes_area < img_area*0.01] = 0
            discard_mask = discard_mask.astype(bool)

            return scores[discard_mask], bboxes[discard_mask]

        return prune_bboxes_1(scores, bboxes)

    def _visualize(self, image, bboxes, scores):
        if bboxes.size == 0:
            return image
        labels = [f' {score:.3f}' for score in scores]
        bboxes = torch.from_numpy(bboxes)
        result_img = torchvision.utils.draw_bounding_boxes(image, 
                                                    boxes=bboxes,
                                                    labels=labels,
                                                    colors=["green"] * len(labels), 
                                                    width=2)
        
        return result_img
    
class VTMUCH(nn.Module):
    def __init__(self, args=None, train_loader=None):
        super(VTMUCH, self).__init__()
        self.args = args
        self._clip = load_download_clip(args.clip_path)
        self._clip = self._clip.to(args.rank).float()
        self._clip_infer = load_download_clip(args.clip_path)
        self._clip_infer = self._clip_infer.to(args.rank).float()
        for param in self._clip_infer.parameters():
            param.requires_grad = False
        self._clip_infer.eval()

        self.args.embed_dim = 512   # CLIP embedding dim

        ### hash layers ###
        self.imgNet = ImgNet(self.args.k_bits, self.args.embed_dim).to(args.rank).float()
        self.txtNet = TxtNet(self.args.k_bits, self.args.embed_dim).to(args.rank).float()

        if self.args.is_train:
            self.object_detection = ObjectDetection(args, self._clip_infer, train_loader)

            self.obj_img_features = self.object_detection.get_obj_clip_features() 

            self.img_objs_clusters = self._make_clusters(torch.cat(list(self.obj_img_features.values()), dim=0)) # ~6secs

            with torch.no_grad():
                self.all_imgs_embed, SI_cont, self.ST, self.SI, self.SF = self._preprocess(train_loader)

            self.SF_comm, self.pseudo_labels, _ = self._recnstrc(self.SF)

            ### densify hash centers by placing one new center between every two old centers ###
            _, _, self.SF_re = self._get_more_centers(self.all_imgs_embed, self.pseudo_labels)
            self.SF = self._refine_SF(self.SF, self.SF_re, SI_cont)

    def set_train(self):
        self._clip.train()
        self.imgNet.train()
        self.txtNet.train()

    def set_eval(self):
        self._clip.eval()
        self.imgNet.eval()
        self.txtNet.eval()

    def _make_clusters(self, feas, method='KMeans'):
        if method == 'KMeans':
            I_clusters = KMeans(n_clusters=self.args.n_clusters, 
                                init='random', 
                                random_state=self.args.seed, 
                                n_init="auto").fit(feas.detach().cpu().numpy())
        else:
            raise NotImplementedError

        return I_clusters

    def _get_hadamard_hash_targets(self, n_class, bit):
            # assert 2*bit > n_class          # depends on # of labels
            assert (bit & (bit-1) == 0)     # bit is a power of 2

            H_K = hadamard(bit)
            H_2K = np.concatenate((H_K, -H_K), 0)
            hash_targets = torch.from_numpy(H_2K[:n_class]).float()

            return hash_targets

    def _refine_SF(self, SF, SF_re, SI_cont):
        normalize = lambda x: 2*(x - torch.amin(x, 1, keepdim=True)) / \
                    (torch.amax(x, 1, keepdim=True) - torch.amin(x, 1, keepdim=True)) - 1
        SI_cont = normalize(SI_cont)

        SF_new = torch.zeros_like(SF)
        SF_new[torch.logical_and(SF == 1., SF_re == 1.)] = 1.
        SF_new[torch.logical_and(SF == -1., SF_re == -1.)] = -1.

        SF_new = torch.where(SF_new == 0., SI_cont, SF_new)

        return SF_new

    def _get_more_centers(self, x, labels, iteration=1):
        ### get more centroids based on centroids obtained from embeds within each commnunity ##
        x_labeled = [x[torch.where(labels == idx)] for idx in labels.unique()]
        x_centroids = torch.stack([comm.mean(dim=0) for comm in x_labeled])
        
        x_centroids_new = torch.stack([(x_centroids[i] + x_centroids[j]) / 2 \
                            for i in range(len(x_centroids)) for j in range(i+1, len(x_centroids))])
        x_centroids_new = torch.cat([x_centroids, x_centroids_new], dim=0)

        ### reassign datapoints to controids based on euclidean distance ###
        dist = torch.cdist(x, x_centroids_new, p=2)
        labels_new = dist.argmin(dim=1)

        ### calculate centroids, re-assign labels (like KMeans) ###
        for _ in range(iteration):
            x_labeled = [x[torch.where(labels_new == idx)] for idx in labels_new.unique()]
            x_centroids_new = torch.stack([comm.mean(dim=0) for comm in x_labeled])
            dist = torch.cdist(x, x_centroids_new, p=2)
            labels_new = dist.argmin(dim=1)

        sim_mat_prime = torch.tile(labels_new, (len(labels_new), 1))
        sim_mat_prime = torch.where((sim_mat_prime.T == sim_mat_prime), 1., -1.)

        x_centroids_new = self._get_hadamard_hash_targets(len(x_centroids_new), self.args.k_bits).to(self.args.rank)
        return x_centroids_new, labels_new, sim_mat_prime

    def _recnstrc(self, sim_mat, min_comm_size=0, beta=None):
        def get_hash_targets(n_class, bit):
            # assert 2*bit > n_class          # depends on # of labels
            assert (bit & (bit-1) == 0)     # bit is a power of 2

            H_K = hadamard(bit)
            H_2K = np.concatenate((H_K, -H_K), 0)
            hash_targets = torch.from_numpy(H_2K[:n_class]).float()

            return hash_targets
    
        convert2tensor = False
        if torch.is_tensor(sim_mat):
            convert2tensor = True
            sim_mat = sim_mat.cpu().numpy()

        # construct graph
        rows, cols = np.where(sim_mat > 0.)
        edges = zip(rows.tolist(), cols.tolist())
        graph = nx.Graph()
        graph.add_edges_from(edges)

        # compute communities
        if beta == None:
            communities = cdlib.algorithms.leiden(graph).communities
        else:
            # g = igraph.Graph.from_networkx(graph)
            # partition = leidenalg.CPMVertexPartition(g, resolution_parameter = beta)
            # optimiser = leidenalg.Optimiser()
            # result = optimiser.optimise_partition(partition)
            # communities = partition
            raise NotImplementedError

        if min_comm_size != 0:
            communities = [comm for comm in communities if len(comm) >= min_comm_size]

        pseudo_labels = torch.full((1, len(sim_mat)), -1, dtype=torch.int64).squeeze().to(self.args.rank)

        for comm_idx, comm in enumerate(communities):
            for i in comm:
                pseudo_labels[i] = comm_idx
        pseudo_labels[pseudo_labels == -1] = len(communities)
        sim_mat_prime = torch.tile(pseudo_labels, (len(pseudo_labels), 1))
        sim_mat_prime = (sim_mat_prime.T == sim_mat_prime)

        if (pseudo_labels == -1).any():
            hash_centers = get_hash_targets(len(communities) + 1, self.args.k_bits).to(self.args.rank)
        else:
            hash_centers = get_hash_targets(len(communities), self.args.k_bits).to(self.args.rank)

        if convert2tensor:
            return sim_mat_prime.float().to(self.args.rank), pseudo_labels, hash_centers
        else:
            return sim_mat_prime.cpu().numpy(), pseudo_labels, hash_centers

    def _list2tensor(self, list_, factor_=1):
        if isinstance(list_, list):
            max_len = max([len(o) for o in list_]) // factor_
        elif isinstance(list_, dict):
            max_len = max([len(o) for o in list_.values()]) // factor_
        else:
            raise ValueError

        tensor_ = torch.zeros((len(list_), max_len, list_[0].shape[-1]))    # [n_imgs, m_objs, embed_dim]
        for i in range(len(list_)):
            n = min(len(list_[i]), max_len)
            tensor_[i, :n, :] = list_[i][:n, :]

        return tensor_
    
    def _batch_compute_pairwise_cos_sim(self, X_batch):
        B = len(X_batch)
        X_batch_sim = torch.zeros(B, B).to(self.args.rank)

        X_batch = torch.nn.functional.normalize(X_batch, dim=-1)

        for i, X_i in tqdm(enumerate(X_batch), desc=f"computing batched cosine similarity"):
            X_i = X_i.repeat(B, 1, 1)
            X_i_sim = torch.bmm(X_i, X_batch.transpose(-1, -2))   # sim(objs in img i -> all objs in all imgs)
            
            X_ij_sim_max = X_i_sim.amax(dim=-2)     # max(sim(objs in img i -> objs in img j))
            X_ji_sim_max = X_i_sim.amax(dim=-1)     # max(sim(objs in img j -> objs in img i))
            n = X_ij_sim_max.count_nonzero(dim=-1) + X_ji_sim_max.count_nonzero(dim=-1)
            X_i_sim = (X_ij_sim_max.sum(dim=-1) + X_ji_sim_max.sum(dim=-1)) / n     # shape = [batch size]

            X_batch_sim[i, :] =  X_i_sim    # X_batch_sim[i] is the similarity between X_i and all other images

        X_batch_sim = torch.nn.functional.normalize(X_batch_sim, dim=-1)
        triu_mask = torch.triu(torch.ones(X_batch_sim.shape), 1).bool()
        std, mean = torch.std_mean(X_batch_sim[triu_mask])

        return X_batch_sim, std, mean
    
    def _preprocess(self, data_loader):
        n = len(data_loader.dataset)
        all_words_embed = {}
        all_imgs_embed = torch.empty(n, self.args.embed_dim, dtype=torch.float).to(self.args.rank)
        all_captions_embed = torch.empty(n, self.args.embed_dim, dtype=torch.float).to(self.args.rank)

        for image, caption, caption_key_padding_mask, index in tqdm(data_loader, 
                                                        desc=f"preprocessing training data"):
            index = index.numpy()
            image = image.float().to(self.args.rank, non_blocking=True)
            caption = caption.to(self.args.rank, non_blocking=True)
            caption_key_padding_mask = caption_key_padding_mask.to(self.args.rank, non_blocking=True)

            img_embed = self._clip_infer.encode_image(image)
            all_imgs_embed[index, :] = img_embed

            caption_embed = self._clip_infer.encode_text(caption, caption_key_padding_mask)
            all_captions_embed[index, :] = caption_embed

            ### get embeddings for each word in caption using CLIP text-encoder
            EOS_TOKEN = caption[0].amax(dim=-1).item()
            START_TOKEN = caption[0, 0].item()
            caption[torch.arange(caption.shape[0]), caption.argmax(dim=-1)] = 0     # remove EOS token
            caption = caption[:, 1:]    # remove start of the sequence token
            for i, words in enumerate(caption):
                tokens = words[words != 0]
                words = torch.zeros(len(tokens), 3, dtype=words.dtype, device=words.device)
                words[:, 0] = START_TOKEN
                words[:, 1] = tokens
                words[:, 2] = EOS_TOKEN
                key_padding_mask = (words == 0)

                words_embed = self._clip_infer.encode_text(words, key_padding_mask)
                all_words_embed[index[i]] = words_embed

        ###########################################################################################
        ### construct SI
        ###########################################################################################
        obj_img_features_trunc = self._list2tensor(self.obj_img_features, self.args.truncation_factor).to(self.args.rank)
        all_obj_sim, all_obj_sim_std, all_obj_sim_mean = self._batch_compute_pairwise_cos_sim(obj_img_features_trunc)
        SI = all_obj_sim.clone()
        SI = torch.where(all_obj_sim > (all_obj_sim_mean + self.args.si_std*all_obj_sim_std), 1., -1.)

        ###########################################################################################
        ### construct ST
        ###########################################################################################
        detected_objs = self.obj_img_features
        text_matched_objs = []
        text_matched_objs_sim = torch.zeros(len(all_words_embed), dtype=torch.float32).to(self.args.rank)
        for i in range(len(all_words_embed)):
            o = detected_objs[i]
            t = all_words_embed[i]
            t2o_sim = calc_pairwise_cos_sim(t, o)   # [m, n] where m is # of word, and n # of objs

            t2o_max, t2o_max_idx = torch.max(t2o_sim, dim=1)
            matched_objs_unique = o[t2o_max_idx.unique(), :]
            text_matched_objs.append(matched_objs_unique.cpu().numpy())
            text_matched_objs_sim[i] = t2o_max.mean()

        matched_objs_labels = [self.img_objs_clusters.predict(Os) for Os in text_matched_objs]

        ST = np.zeros((n, n))
        for i in tqdm(range(n), desc="constructing ST"):
            for j in range(i, n):
                ST[i, j] = np.isin(matched_objs_labels[i], matched_objs_labels[j]).any()
        ST = ST + np.triu(ST, 1).T
        ST[ST == 0] = -1.
        ST = torch.from_numpy(ST).float().to(self.args.rank)

        ###########################################################################################
        ### construct SF
        ###########################################################################################
        SF = torch.where(torch.logical_or(ST > 0, SI > 0), 1., -1.)
        
        return all_imgs_embed, all_obj_sim, ST, SI, SF
        
    def forward(self, index, image, key_padding_mask, caption):
        assert not self._clip_infer.training
        output_dict = {}

        with torch.no_grad():
            img_feature = self._clip_infer.encode_image(image)

        txt_feature = self._clip.encode_text(caption, key_padding_mask)
        txt_hash = self.txtNet(txt_feature)
        img_hash = self.imgNet(img_feature)

        output_dict['txt_hash'] = txt_hash
        output_dict['img_hash'] = img_hash

        if self._clip.training:
            col, row = np.meshgrid(index.numpy(), index.numpy())
            output_dict['SF'] = self.SF[row, col]

        return output_dict

