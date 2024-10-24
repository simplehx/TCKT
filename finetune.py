# coding:utf-8
import logging
import datetime
import math
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
import pandas as pd
import random

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FinetuneDataset(Dataset):
    def __init__(self, dataset, pretrain_poi2path, finetune_poi2path, constraint_dict, layer_count, user_align, stage):
        super().__init__()
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.pretrain_poi2path = pretrain_poi2path
        self.finetune_poi2path = finetune_poi2path
        self.constraint_dict = constraint_dict
        self.layer_count = layer_count
        self.stage = stage
        self.domain_dict = {"c": [1], "p": [0]}
        self.user_align = user_align

    def __getitem__(self, index):
        user_id, poi_list, poi_label, data_type = self.dataset[index]
        if data_type == "c":
            poi_list = [self.finetune_poi2path[poi] for poi in poi_list]
            poi_label = self.finetune_poi2path[poi_label]
            user = torch.LongTensor([self.user_align[user_id]])
        elif data_type == "p":
            poi_list = [self.pretrain_poi2path[poi] for poi in poi_list]
            poi_label = self.pretrain_poi2path[poi_label]
            user = torch.LongTensor([user_id])
            
        layer_mask_accurate = []
        poi_label_str = list(map(str, poi_label))
        for i in range(len(poi_label_str)):
            if i == 0:
                layer_list = self.constraint_dict["-1"]
            else:
                layer_list = self.constraint_dict["_".join(poi_label_str[: i])]
            layer_mask = torch.zeros(self.layer_count[i], dtype=torch.long)
            layer_mask[layer_list] = 1
            layer_mask_accurate.append(layer_mask)     
        poi_label = torch.LongTensor(poi_label)
        poi_list = torch.LongTensor(poi_list)
        
        domain = torch.FloatTensor(self.domain_dict[data_type])
        data_type = torch.LongTensor([0 if data_type == "c" else 1])

        data = {"domain": domain, "user": user, "poi": poi_list, "poi_label": poi_label, "layer_mask_accurate": layer_mask_accurate, "data_type": data_type}
        return data

    def __len__(self):
        return self.dataset_len

class FeatureExtractorCold(nn.Module):
    def __init__(self, layer_count, hidden, n_layers, attn_heads, seq_len, dropout=0.1):
        super(FeatureExtractorCold, self).__init__()
        self.seq_len = seq_len
        self.poi_embedding_layer = POIEmbedding(hidden, layer_count)
        self.user_embedding_layer = UserEmbedding(hidden, user_count)
        self.dt_embedding_layer = TemporalEmbedding(hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=attn_heads, dim_feedforward=hidden*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=n_layers)
        
        self.layer_count = layer_count
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.LayerNorm(1)
        )
        self.attention_layer_norm = nn.LayerNorm(hidden)
        self.user_layer_norm = nn.LayerNorm(hidden)

    def forward(self, data, stage):
        if stage == "pretrain":
            select_data = data["poi"][data["data_type"].squeeze(1) == 1]
        else:
            select_data = data["poi"]
        traj_embedding = self.poi_embedding_layer(select_data)
        user_embedding = self.user_embedding_layer(data["user"]).squeeze(1)
        encoder_output = self.transformer_encoder(traj_embedding)
        weight = F.softmax(self.attention_pooling(encoder_output), dim=1)
        model_output = torch.sum(weight * encoder_output, dim=1)
        model_output = self.attention_layer_norm(model_output)
        model_output = self.user_layer_norm(model_output + user_embedding)
        return model_output

class LabelPredictor(nn.Module):
    def __init__(self, layer_count, hidden, poi_embedding):
        super().__init__()
        self.layer_count = layer_count
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

    def forward(self, out_feature, layer_mask_accurate, poi_label, tf_rate, stage):
        label_split_list = torch.chunk(poi_label, dim=-1, chunks=len(self.layer_count))

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
                test_constraint = test_constraint_list[i].expand_as(layer_out)
                layer_out = layer_out.masked_fill(test_constraint.eq(0), -1e4)
            layer_out_list.append(layer_out)
            label_index = label_split_list[i].squeeze(-1)
            layer_max_embedding = self.poi_embedding.emb_layer_list[i](label_index)

            linear_hid = linear_hid + layer_max_embedding + identity
        
        return layer_out_list

class TemporalEmbedding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        month_pad, day_pad, weekday_pad, hour_pad, min_pad = 0, 0, 0, 0, 0
        self.month_emb = nn.Embedding(12 + 1, embed_size, padding_idx=month_pad) # 1, 2, 3, ..., 12    padding=0
        self.day_emb = nn.Embedding(31 + 1, embed_size, padding_idx=day_pad) # 1, 2, 3, ..., 31    padding=0
        self.weekday_emb = nn.Embedding(7 + 1, embed_size, padding_idx=weekday_pad) # 1, 2, 3, ..., 7    padding=0
        self.hour_emb = nn.Embedding(24 + 1, embed_size, padding_idx=hour_pad) # 0, 1, 2, ..., 23    padding=0
        self.min_emb = nn.Embedding(6 + 1, embed_size, padding_idx=min_pad) # 0, 1, 2, 3, 4, 5    padding=0
    
    def forward(self, x):
        month, day, weekday, hour, min = torch.chunk(x, dim=2, chunks=5)
        month, day, weekday, hour, min = month.squeeze(2), day.squeeze(2), weekday.squeeze(2), hour.squeeze(2), min.squeeze(2)

        month_embedding = self.month_emb(month)
        day_embedding = self.day_emb(day)
        weekday_embedding = self.weekday_emb(weekday)
        hour_embedding = self.hour_emb(hour)
        min_embedding = self.min_emb(min)
        return (month_embedding + day_embedding + weekday_embedding + hour_embedding + min_embedding) / 5

class FeatureExtractorPretrain(nn.Module):
    def __init__(self, layer_count, hidden, n_layers, attn_heads, seq_len, dropout=0.1):
        super(FeatureExtractorPretrain, self).__init__()
        self.seq_len = seq_len
        self.poi_embedding_layer = POIEmbedding(hidden, layer_count)
        self.user_embedding_layer = UserEmbedding(hidden, user_count)
        self.dt_embedding_layer = TemporalEmbedding(hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=attn_heads, dim_feedforward=hidden*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=n_layers)
        
        self.layer_count = layer_count
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.LayerNorm(1)
        )
        self.attention_layer_norm = nn.LayerNorm(hidden)
        self.user_layer_norm = nn.LayerNorm(hidden)

    def forward(self, data, stage):
        if stage == "pretrain":
            select_data = data["poi"][data["data_type"].squeeze(1) == 1]
        else:
            select_data = data["poi"]
        traj_embedding = self.poi_embedding_layer(select_data)
        user_embedding = self.user_embedding_layer(data["user"]).squeeze(1)
        encoder_output = self.transformer_encoder(traj_embedding)
        weight = F.softmax(self.attention_pooling(encoder_output), dim=1)
        model_output = torch.sum(weight * encoder_output, dim=1)
        model_output = self.attention_layer_norm(model_output)
        model_output = self.user_layer_norm(model_output + user_embedding)
        
        return model_output

def finetune_metrics(out_list, poi_label):
    out_list = [torch.softmax(out, dim=-1).log() for out in out_list]
    k_list = [out.size(-1) if out.size(-1) < 10 else 10 for out in out_list]
    padding_tuple = tuple([-1 for _ in range(len(k_list))])
    topk_list = []
    for topk, out in zip(k_list, out_list):
        out_value, out_index = torch.topk(out, dim=1, k=topk)
        topk_list.append([out_value, out_index])

    b_size = poi_label.size(0)
    
    result = None
    for index, (out_value, out_index) in enumerate(topk_list):
        if result is None:
            result = out_value
        else:
            result = (result + out_value.unsqueeze(1)).view(b_size, -1)
        
        if index != len(k_list) - 1:
            result = result.unsqueeze(2)

    path_index = torch.cartesian_prod(*[torch.arange(topk) for topk in k_list]).cuda()
    k = 3000 if result.size(1) >= 3000 else 1000
    result_value_top, result_index_top = torch.topk(result, dim=1, k=k)
    temp = path_index[result_index_top]
    layer_out_list = []
    for index, (out_value, out_index) in enumerate(topk_list):
        layer_out = out_index[torch.arange(b_size).unsqueeze(1), temp[..., index]]
        layer_out_list.append(layer_out)

    r = torch.stack(layer_out_list, dim=2)
    mrr_list = []
    for i in range(r.size(0)):
        result_index = torch.where(torch.all(poi_label[i] == r[i], dim=1))[0]
        rank = result_index.item() if len(result_index) > 0 else r.size(1)
        mrr_list.append(1 / (rank + 1))
            
    top1_result = r[:,:1,:]
    top5_result = r[:,:5,:]
    top10_result = r[:,:10,:]

    batch_total = b_size
    top1_correct = torch.any(torch.all(poi_label.unsqueeze(1) == top1_result, dim=-1), dim=-1).sum().item()
    top5_correct = torch.any(torch.all(poi_label.unsqueeze(1) == top5_result, dim=-1), dim=-1).sum().item()
    top10_correct = torch.any(torch.all(poi_label.unsqueeze(1) == top10_result, dim=-1), dim=-1).sum().item()
    return batch_total, top1_correct, top5_correct, top10_correct, mrr_list

class POIEmbedding(nn.Module):
    def __init__(self, embedding_dim, layer_count_list):
        super(POIEmbedding, self).__init__()
        self.layer_count_list = layer_count_list
        self.emb_layer_list = nn.ModuleList(
            [nn.Embedding(num_embeddings=layer_count+1, embedding_dim=embedding_dim, padding_idx=layer_count) for layer_count in layer_count_list]
        )
    def forward(self, poi_path):
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

class FinetuneLoss(nn.Module):
    def __init__(self, layer_count_list):
        super().__init__()
        self.layer_count_list = layer_count_list
        self.layer_loss_fn = nn.CrossEntropyLoss()
    def forward(self, out_list, poi_label):
        label_split_list = torch.chunk(poi_label, dim=1, chunks=len(self.layer_count_list))
        loss_result = None
        for index, (out, label_split) in enumerate(zip(out_list, label_split_list)):
            layer_loss = self.layer_loss_fn(out, label_split.squeeze(1))
            if loss_result is None:
                loss_result = layer_loss
            else:
                loss_result += layer_loss
        return loss_result / len(label_split_list)

class DomainClassifier(nn.Module):
    def __init__(self, hidden):
        super(DomainClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Linear(hidden, 1)
        )

    def forward(self, input):
        out = self.layer(input)
        return out

def dt_converter(*args):
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return now.timetuple()

class Trainner:
    def __init__(self, train_data, test_data):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.label_loss_fn = FinetuneLoss(layer_count)
        self.domain_loss_fn = nn.BCEWithLogitsLoss()
        self.distance_loss_fn = nn.CosineEmbeddingLoss()

        seq_len = 5
        hidden = 512
        self.feature_extractor_pretrain = FeatureExtractorPretrain(layer_count=layer_count, hidden=hidden, n_layers=4, attn_heads=4, seq_len=seq_len).to(self.device)
        self.feature_extractor_cold = FeatureExtractorCold(layer_count=layer_count, hidden=hidden, n_layers=4, attn_heads=4, seq_len=seq_len).to(self.device)
        self.label_predictor = LabelPredictor(layer_count=layer_count, hidden=hidden, poi_embedding=self.feature_extractor_cold.poi_embedding_layer).to(self.device)
        self.domain_classifier = DomainClassifier(hidden).to(self.device)

        self.load_checkpoints(checkpoint_epoch=checkpoint_epoch)

        params_to_optimize = list(self.feature_extractor_cold.parameters())
        for name, param in self.label_predictor.named_parameters():
            if "poi_embedding" not in name:
                params_to_optimize.append(param)

        self.optimizer_C = torch.optim.AdamW(params_to_optimize, lr=1e-4, weight_decay=1e-3)
        self.optimizer_D = torch.optim.AdamW(self.domain_classifier.parameters(), lr=1e-4, weight_decay=1e-3)
        self.scheduler_C = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer_C, step_size=5, gamma=0.8)
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer_D, step_size=5, gamma=0.8)

        self.scaler = GradScaler()
        self.tf_rate = 0.9
        self.tf_step = 0

        batch_size = 500
        train_dataset = FinetuneDataset(train_data, pretrain_poi2path, finetune_poi2path, constraint_dict, layer_count, user_align, stage="train")
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        test_dataset = FinetuneDataset(test_data, pretrain_poi2path, finetune_poi2path, constraint_dict, layer_count, user_align, stage="test")
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    def load_checkpoints(self, checkpoint_epoch):
        checkpoint = torch.load(f"./save/{dir_name}/best_model_{checkpoint_epoch}.pth")["model"]

        feature_extractor_keys = set(self.feature_extractor_pretrain.state_dict())
        match_dict = dict()
        for key, value in checkpoint.items():
            key = key.replace("model1.", "")
            if key in feature_extractor_keys:
                match_dict[key] = value
        self.feature_extractor_pretrain.load_state_dict(match_dict, strict=False)

        for key, param in self.feature_extractor_pretrain.named_parameters():
            param.requires_grad = False
        
    def iteration(self, data_loader, is_train=True, mode=""):
        reciprocal_rank = []
        epoch_total, epoch_top1_correct, epoch_top5_correct, epoch_top10_correct  = 0, 0, 0, 0
        
        running_domain_loss, running_F_loss, running_cosine_loss = .0, .0, .0
        lamb = 0.1
        for i, data in enumerate(tqdm(data_loader)):
            data = {key: value.to(device) if not isinstance(value, list) else [mask.to(device) for mask in value] for key, value in data.items()}
            if is_train:
                with autocast():
                    feature_cold = self.feature_extractor_cold(data, "other")
                    domain_out = self.domain_classifier(feature_cold.detach())
                    domain_loss = self.domain_loss_fn(domain_out, data["domain"])
                    running_domain_loss += domain_loss.item()
                    self.scaler.scale(domain_loss).backward()
                    self.scaler.step(self.optimizer_D)

                    feature_pretrain = self.feature_extractor_pretrain(data, "other")
                    distance_loss = self.distance_loss_fn(feature_cold, feature_pretrain, torch.ones(feature_cold.size(0)).to(self.device))
                    running_cosine_loss += distance_loss.item()
                    self.scaler.scale(distance_loss).backward(retain_graph=True)
                    
                    class_out = self.label_predictor(feature_cold, data["layer_mask_accurate"], data["poi_label"], self.tf_rate, "train")

                    domain_out = self.domain_classifier(feature_cold)
                    loss = self.label_loss_fn(class_out, data["poi_label"]) - lamb * self.domain_loss_fn(domain_out, data["domain"])
                    running_F_loss += loss.item()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer_C)

                    self.optimizer_D.zero_grad()
                    self.optimizer_C.zero_grad()
                    
                    self.scaler.update()
            else:
                with torch.no_grad():
                    with autocast():
                        feature = self.feature_extractor_cold(data, "other")
                        out_list = self.label_predictor(feature, data["layer_mask_accurate"], data["poi_label"], self.tf_rate, "test")
                        batch_total, batch_top1_correct, batch_top5_correct, batch_top10_correct, batch_mrr_list = finetune_metrics(out_list, data["poi_label"])
                    epoch_total += batch_total
                    epoch_top1_correct += batch_top1_correct
                    epoch_top5_correct += batch_top5_correct
                    epoch_top10_correct += batch_top10_correct
                    reciprocal_rank.extend(batch_mrr_list)
                
        if is_train:
            return {"domain": running_domain_loss / (i+1), "cosine": running_cosine_loss / (i+1), "F": running_F_loss / (i+1)}
        else:
            return {"acc1": epoch_top1_correct / epoch_total, "acc5": epoch_top5_correct / epoch_total, "acc10": epoch_top10_correct / epoch_total, "mrr": np.mean(reciprocal_rank)}
    def train(self):
        if self.tf_step >= 6:
            self.tf_rate = self.tf_rate * 0.9
            self.tf_step = 0
        self.tf_step += 1

        result = self.iteration(self.train_loader, is_train=True)
        return result

    def test(self):
        result = self.iteration(self.test_loader, is_train=False)
        return result

    def start(self):
        for epoch in range(30):
            train_result = self.train()
            logging.info(f"{epoch}" + " | ".join([f"{key}: {value}" for key, value in train_result.items()]))
            test_result = self.test()
            logging.info(f"{epoch}" + " | ".join([f"{key}: {value}" for key, value in test_result.items()]))
            logging.info("-" * 100)


if __name__ == '__main__':
    checkpoint_epoch = 180
    dataset_name = "gowalla"
    cold_count = 5
    km_list = [50, 1, 0.05]
    dir_name = "_".join(map(str, km_list))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.Formatter.converter = dt_converter
    logging.basicConfig(filename=f"./save/{dir_name}/{dataset_name}.log", level=logging.INFO, format=logging_format, datefmt=date_format)
        
    dataset = np.load(f"./cold_dataset/{dataset_name}/cold_dataset_{cold_count}_finetune.npy", allow_pickle=True).tolist()
    train_dataset, test_dataset, pretrain_dataset = dataset["train"], dataset["test"], dataset["pretrain"]
    train_dataset.extend(pretrain_dataset)
    all_pretrain_label = set([data[2] for data in pretrain_dataset])
    
    user_align = np.load(f"./align_data/finetune/{dataset_name}_user_align.npy", allow_pickle=True).tolist()
    path_name_list = [f"path{i+1}" for i in range(len(km_list)+1)]
    poi_trans_dict = np.load(f"./align_data/{dir_name}/finetune/{dataset_name}/poi_match_dict.npy", allow_pickle=True).tolist()
    pretrain_poi2path = np.load(f"./align_data/{dir_name}/pretrain/foursquare_poi2path.npy", allow_pickle=True).tolist()
    finetune_path_df = [[finetune_poi] + pretrain_poi2path[pretrain_poi] for pretrain_poi, finetune_poi in poi_trans_dict.items()]
    finetune_path_df.extend([[-1] + pretrain_poi2path[poi] for poi in all_pretrain_label])
    finetune_path_df = pd.DataFrame(finetune_path_df, columns=["poi"] + path_name_list)
    finetune_poi2path = {finetune_poi: tuple(pretrain_poi2path[pretrain_poi]) for pretrain_poi, finetune_poi in poi_trans_dict.items()}
    
    pretrain_param = np.load(f"./align_data/{dir_name}/pretrain/pretrain_param.npy", allow_pickle=True).tolist()
    user_count = pretrain_param["user_count"]
    layer_count = pretrain_param["layer_count"]

    group_by_list = [path_name_list[: i + 1] for i in range(len(path_name_list) - 1)]

    constraint_path = f"./align_data/{dir_name}/finetune/{dataset_name}/{cold_count}_constraint_dict.npy"
    constraint_dict = np.load(constraint_path, allow_pickle=True).tolist()
    
    test_constraint_list = []
    for i, path_name in enumerate(path_name_list):
        layer_list = list(set(finetune_path_df[path_name]))
        layer_mask = torch.zeros(layer_count[i], dtype=torch.long).to(device)
        layer_mask[layer_list] = 1
        test_constraint_list.append(layer_mask)
    
    trainner = Trainner(train_dataset, test_dataset)
    trainner.start()