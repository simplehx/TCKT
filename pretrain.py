# coding:utf-8

import os
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import lmdb
import pickle
import torch
from functools import reduce
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
import logging
import datetime
model_save_base = "./save/"
km_list = [50, 1, 0.05]
dir_name = "_".join(map(str, km_list))
dataset_name = "gowalla"
model_save_path = os.path.join(model_save_base, dir_name)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

pretrain_data_path = "./pretrain/pretrain_dataset"

class PretrainDataset(Dataset):
    def __init__(self, txn, dataset_index_list, poi2_path_dict, concat_tree_dict, constraint_dict, layer_count):
        super().__init__()
        self.txn = txn
        self.dataset_index_list = dataset_index_list
        self.dataset_len = len(dataset_index_list)
        self.pos_count = 20
        self.neg_count = 20
        self.sample_count = self.neg_count * 2
        self.poi2_path_dict = poi2_path_dict
        self.concat_tree_dict = concat_tree_dict
        self.all_poi_list = np.array(list(poi2_path_dict))
        self.all_path = set(poi2_path_dict.values())
        self.all_poi_list = np.array(list(poi2_path_dict))
        self.constraint_dict = constraint_dict
        self.layer_count = layer_count
        self.tree_padding = tuple(layer_count)
    
    
    def __getitem__(self, index):
        layers = len(layer_count)
        dataset_index = self.dataset_index_list[index]
        user, traj_seq, traj_label = pickle.loads(self.txn.get(dataset_index.encode()))
        user_traj_pos = traj_seq + [traj_label]
        user_traj_neg = list(set(np.random.choice(self.all_poi_list, self.sample_count)).difference(set(user_traj_pos)))[:self.neg_count]

        label_path = list(self.poi2_path_dict[traj_label])
        lase_path_set = set(reduce(lambda d, k: d[k], label_path[:-1], self.concat_tree_dict))
        pos_path_list = [tuple(label_path[:-1] + [path5_]) for path5_ in lase_path_set.difference(set([label_path[-1]]))]
        
        label_pos_mask = [1 for _ in range(self.pos_count)]
        if len(pos_path_list) < self.pos_count:
            pos_path_list_ = [tuple(label_path[:layers-2] + [path4_, path5_]) for path4_, path5_dict in reduce(lambda d, k: d[k], label_path[:layers-2], self.concat_tree_dict).items() for path5_ in path5_dict if label_path[-2] != path4_]
            pos_path_list += pos_path_list_
            if len(pos_path_list) < self.pos_count:
                label_pos = pos_path_list + [self.tree_padding for _ in range(self.pos_count - len(pos_path_list))]
                label_pos_mask = [1 for _ in range(len(pos_path_list))] + [0 for _ in range(self.pos_count - len(pos_path_list))]
            else:
                label_pos = random.sample(pos_path_list, self.pos_count)
        else:
            label_pos = random.sample(pos_path_list, self.pos_count)

        label_neg = list(set(np.random.choice(self.all_poi_list, self.sample_count)).difference(set([traj_label])))[:self.neg_count]
        label_neg = [self.poi2_path_dict[label] for label in label_neg]

        traj_seq = [self.poi2_path_dict[poi] for poi in traj_seq]
        traj_label = self.poi2_path_dict[traj_label]
        user_traj_pos = traj_seq + [traj_label]
        user_traj_neg = [self.poi2_path_dict[poi] for poi in user_traj_neg]


        layer_mask_list = []
        poi_label_str = list(map(str, traj_label))
        for i in range(len(poi_label_str)):
            if i == 0:
                continue
            layer_list = self.constraint_dict["_".join(poi_label_str[: i])]
            layer_mask = torch.zeros(self.layer_count[i], dtype=torch.long)
            layer_mask[layer_list] = 1
            layer_mask_list.append(layer_mask)
        
        user_tensor = torch.LongTensor([user])
        traj_tensor = torch.LongTensor(traj_seq)
        label_tensor = torch.LongTensor(traj_label)
        user_traj_pos_tensor = torch.LongTensor(user_traj_pos)
        user_traj_neg_tensor = torch.LongTensor(user_traj_neg)
        label_pos_tensor = torch.LongTensor(label_pos)
        label_neg_tensor = torch.LongTensor(label_neg)
        label_pos_mask = torch.LongTensor(label_pos_mask)

        data = {"user": user_tensor, "traj": traj_tensor, "label": label_tensor, "traj_pos": user_traj_pos_tensor, 
                "traj_neg": user_traj_neg_tensor, "label_pos": label_pos_tensor, "label_pos_mask": label_pos_mask,
                "label_neg": label_neg_tensor, "layer_mask_list": layer_mask_list}
        return data

    def __len__(self):
        return self.dataset_len

class POIEmbedding(nn.Module):
    def __init__(self, embedding_dim, layer_count_list):
        super(POIEmbedding, self).__init__()
        self.layer_count_list = layer_count_list
        self.emb_layer_list = nn.ModuleList(
            [nn.Embedding(num_embeddings=layer_count + 1, embedding_dim=embedding_dim, padding_idx=layer_count) for layer_count in layer_count_list]
        )
        self.gru = nn.LSTM(embedding_dim, embedding_dim, num_layers=1, batch_first=True)

    def forward(self, poi_path, type_="traj"):
        poi_path_split_list = torch.chunk(poi_path, dim=-1, chunks=len(self.layer_count_list))
        emb_result = None
        for index, poi_path_split in enumerate(poi_path_split_list):
            emb = self.emb_layer_list[index](poi_path_split.squeeze(-1))
            if emb_result is None:
                emb_result = emb
            else:
                emb_result += emb
        return emb_result / len(poi_path_split_list)

class UserEmbedding(nn.Module):
    def __init__(self, embedding_dim, user_count):
        super().__init__()
        self.user_emb = nn.Embedding(user_count, embedding_dim)

    def forward(self, x):
        return self.user_emb(x)

class Model1(nn.Module):
    def __init__(self, seq_len, layer_count, hidden, n_layers, attn_heads, dropout, poi_embedding):
        super().__init__()
        self.seq_len = seq_len
        self.layer_count = layer_count
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=attn_heads, dim_feedforward=hidden*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=n_layers)

        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.LayerNorm(1)
        )

        self.attention_layer_norm = nn.LayerNorm(hidden)
        self.user_layer_norm = nn.LayerNorm(hidden)
        self.poi_embedding = poi_embedding
        linear_layer_list = []
        for i, count in enumerate(layer_count):
            linear = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(inplace=True)
            )
            linear_layer_list.append(linear)
        self.linear_layer_list = nn.ModuleList(linear_layer_list)
        
        self.out_layer_list = nn.ModuleList([nn.Linear(hidden, count) for count in layer_count])
        

    def forward(self, traj_embedding, user_embedding, layer_mask_list, poi_label):
        label_split_list = torch.chunk(poi_label, dim=-1, chunks=len(self.layer_count))
        encoder_output = self.transformer_encoder(traj_embedding)
        weight = F.softmax(self.attention_pooling(encoder_output), dim=1)
        model_output = torch.sum(weight * encoder_output, dim=1)
        model_output = self.attention_layer_norm(model_output)
        out_feature = self.user_layer_norm(model_output + user_embedding)

        linear_hid_list = []
        layer_out_list = []
        identity = None
        for i in range(len(self.layer_count)):
            if i == 0:
                identity = out_feature
                linear_hid = self.linear_layer_list[i](out_feature)
            else:
                identity = linear_hid
                linear_hid = self.linear_layer_list[i](linear_hid)
            linear_hid_list.append(linear_hid)
            layer_out = self.out_layer_list[i](linear_hid)
            if i > 0:
                layer_out = layer_out.masked_fill(layer_mask_list[i-1].eq(0), -1e4)
            layer_out_list.append(layer_out)
            label_index = label_split_list[i].squeeze(-1)
            layer_max_embedding = self.poi_embedding.emb_layer_list[i](label_index)

            linear_hid = linear_hid + layer_max_embedding + identity
        
        return layer_out_list

class Model1Loss(nn.Module):
    def __init__(self, layer_count):
        super().__init__()
        self.layer_count_list = layer_count
        self.layer_loss_fn_list = nn.ModuleList([nn.CrossEntropyLoss() for _ in range(len(layer_count))])


    def forward(self, model1_out, poi_label):
        label_split_list = torch.chunk(poi_label, dim=1, chunks=len(self.layer_count_list))
        loss_result = None
        for index, (out, label_split) in enumerate(zip(model1_out, label_split_list)):
            layer_loss = self.layer_loss_fn_list[index](out, label_split.squeeze(1))
            if loss_result is None:
                loss_result = layer_loss
            else:
                loss_result += layer_loss
        return loss_result / len(label_split_list)

class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, user_embedding, traj_pos_embedding, traj_neg_embedding):
        user_embedding = user_embedding.unsqueeze(1)
        user_pos_score = (user_embedding * traj_pos_embedding).sum(-1)
        user_neg_score = (user_embedding * traj_neg_embedding).sum(-1)
        out = {"pos_score": user_pos_score, "neg_score": user_neg_score}
        return out

class Model3(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, label_embedding, tree_pos_embedding, tree_neg_embedding):
        label_embedding = label_embedding.unsqueeze(1)
        tree_pos_score = (label_embedding * tree_pos_embedding).sum(-1)
        tree_neg_score = (label_embedding * tree_neg_embedding).sum(-1)
        out = {"pos_score": tree_pos_score, "neg_score": tree_neg_score}
        return out

class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    
    def forward(self, data, mask=None):
        pos_score, neg_score = data["pos_score"], data["neg_score"]
        if mask is not None:
            loss = -F.logsigmoid((pos_score.unsqueeze(2) - neg_score.unsqueeze(1))).mean(-1)[mask.eq(1)].mean()
        else:
            loss = -F.logsigmoid((pos_score.unsqueeze(2) - neg_score.unsqueeze(1))).mean()
        return loss

class LossFunction(nn.Module):
    def __init__(self, layer_count):
        super().__init__()
        self.model1_loss = Model1Loss(layer_count)
        self.bpr_loss1 = BPRLoss()
        self.bpr_loss2 = BPRLoss()

    def forward(self, model1_out, model2_out, model3_out, mask, model1_label):
        model1_loss = self.model1_loss(model1_out, model1_label)
        model2_loss = self.bpr_loss1(model2_out)
        model3_loss = self.bpr_loss2(model3_out, mask=mask)

        return model1_loss, model2_loss, model3_loss

class Model(nn.Module):
    def __init__(self, user_count, layer_count, hidden, n_layers, attn_heads, seq_len, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.poi_embedding_layer = POIEmbedding(hidden, layer_count)
        self.user_embedding_layer = UserEmbedding(hidden, user_count)
        self.model1 = Model1(self.seq_len, layer_count, hidden, n_layers, attn_heads, dropout, self.poi_embedding_layer)
        self.model2 = Model2()
        self.model3 = Model3()

    def forward(self, data):
        traj_embedding = self.poi_embedding_layer(data["traj"])
        user_embedding = self.user_embedding_layer(data["user"]).squeeze(1)
        label_embedding = self.poi_embedding_layer(data["label"], type_="label")
        traj_pos_embedding = self.poi_embedding_layer(data["traj_pos"])
        traj_neg_embedding = self.poi_embedding_layer(data["traj_neg"])
        model1_out = self.model1(traj_embedding, user_embedding, data["layer_mask_list"], data["label"])
        model2_out = self.model2(user_embedding, traj_pos_embedding, traj_neg_embedding)

        tree_pos_embedding = self.poi_embedding_layer(data["label_pos"])
        tree_neg_embedding = self.poi_embedding_layer(data["label_neg"])
        model3_out = self.model3(label_embedding, tree_pos_embedding, tree_neg_embedding)

        return model1_out, model2_out, model3_out

def dt_converter(*args):
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return now.timetuple()

if __name__ == '__main__':
    seq_len = 5

    logging_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.Formatter.converter = dt_converter
    logging.basicConfig(filename=f"./save/{dir_name}/train.log", level=logging.INFO, format=logging_format, datefmt=date_format)

    pretrain_env = lmdb.open(pretrain_data_path, readonly=True)
    pretrain_txn = pretrain_env.begin()
    pretrain_index_list = [str(i) for i in range(pretrain_txn.stat()["entries"])]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    constraint_dict = np.load(f"./align_data/{dir_name}/pretrain/constraint_dict.npy", allow_pickle=True).tolist()
    concat_tree_dict = np.load(f"./align_data/{dir_name}/pretrain/foursquare_tree_dict.npy", allow_pickle=True).tolist()
    poi2_path_dict = np.load(f"./align_data/{dir_name}/pretrain/foursquare_poi2path.npy", allow_pickle=True).tolist()
    poi2_path_dict = {poi: tuple(path) for poi, path in poi2_path_dict.items()}
    all_path_set = set(poi2_path_dict.values())

    pretrain_path_set = set(poi2_path_dict.values())

    pretrain_param = np.load(f"./align_data/{dir_name}/pretrain/pretrain_param.npy", allow_pickle=True).tolist()
    user_count = pretrain_param["user_count"]
    layer_count = pretrain_param["layer_count"]

    batch_size = 8000
    pretrain_dataset = PretrainDataset(pretrain_txn, pretrain_index_list, poi2_path_dict, concat_tree_dict, constraint_dict, layer_count)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    hidden = 512
    loss_fn = LossFunction(layer_count).to(device)
    
    model = Model(user_count=user_count, layer_count=layer_count, hidden=hidden, n_layers=4, attn_heads=4, seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=6, gamma=0.8)

    scaler = GradScaler()
    best_loss = float("inf")
    model.train()
    for epoch in range(201):
        epoch_loss_list = []
        epoch_loss1_list, epoch_loss2_list, epoch_loss3_list = [], [], []
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        for index, data in enumerate(tqdm(pretrain_loader)):
            optimizer.zero_grad()
            if index != 0 and index % 3000 == 0:
                logging.info(f"epoch {epoch}, current_lr {current_lr}, loss {np.mean(epoch_loss_list)}, loss1 {np.mean(epoch_loss1_list)}, loss2 {np.mean(epoch_loss2_list)}, loss3 {np.mean(epoch_loss3_list)}")
            with autocast():
                data = {key: value.to(device) if not isinstance(value, list) else [mask.to(device) for mask in value] for key, value in data.items()}
                model1_out, model2_out, model3_out = model(data)
                loss1, loss2, loss3 = loss_fn(model1_out, model2_out, model3_out, data["label_pos_mask"], data["label"])
                loss = loss1 + loss2 + loss3  # or awl_loss
            if torch.isnan(loss).any():
                logging.info(f"{epoch} | {index} | {loss1}, {loss2}, {loss3}")
                continue
            epoch_loss1_list.append(loss1.item())
            epoch_loss2_list.append(loss2.item())
            epoch_loss3_list.append(loss3.item())
            epoch_loss_list.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        epoch_loss = np.mean(epoch_loss_list)
        scheduler.step()
        if epoch % 5 == 0:
            checkpoint = {
                "epoch": epoch, 
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss
            }
            torch.save(checkpoint, os.path.join(model_save_path, f"best_model_{epoch}.pth"))
        
        logging.info(f"epoch {epoch}, current_lr {current_lr}, loss {np.mean(epoch_loss_list)}, loss1 {np.mean(epoch_loss1_list)}, loss2 {np.mean(epoch_loss2_list)}, loss3 {np.mean(epoch_loss3_list)}")
        logging.info("-" * 100)