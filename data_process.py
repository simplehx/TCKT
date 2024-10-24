# coding:utf-8

import pandas as pd
from tqdm import tqdm
import datetime
from itertools import chain
import random
import calendar
import os
import numpy as np
import lmdb
import pickle
import shutil
from pandarallel import pandarallel
from sklearn.neighbors import KDTree, BallTree
from sklearn.cluster import DBSCAN
from  sklearn.preprocessing import MinMaxScaler
from collections import Counter
from math import cos, sin, atan2, sqrt, radians, degrees
from pandarallel import pandarallel
import multiprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import copy
import itertools
import geopandas as gpd
from shapely.geometry import Point
from joblib import Parallel, delayed
pandarallel.initialize(progress_bar=True)

window = 5

filter_flag = False
cold_count_list = [5, 10, 15, 20]
km_list = [50, 1, 0.05]
dataset_name = "gowalla"
dir_name = "_".join(map(str, km_list))
save_path = f"./align_data/{dir_name}"
pretrain_path = os.path.join(save_path, f"pretrain/{dataset_name}")
finetune_path = os.path.join(save_path, f"finetune/{dataset_name}")
if not os.path.exists(pretrain_path): os.makedirs(pretrain_path)
if not os.path.exists(finetune_path): os.makedirs(finetune_path)

if not os.path.exists("./align_data/pretrain"): os.makedirs("./align_data/pretrain")
if not os.path.exists("./align_data/finetune"): os.makedirs("./align_data/finetune")
if not os.path.exists(f"./cold_dataset/{dataset_name}"): os.makedirs(f"./cold_dataset/{dataset_name}")

world = gpd.read_file("./us/ne_110m_admin_0_countries.shp")
usa = world[world['ADMIN'] == 'United States of America']

def is_usa(data):
    point = Point(data["lng"], data["lat"])
    return usa.contains(point).any()

def get_centroid(cluster):
    x = y = z = 0
    coord_num = len(cluster)
    for coord in cluster:
        lat, lon = radians(coord[0]), radians(coord[1])
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)
    x /= coord_num
    y /= coord_num
    z /= coord_num
    lat_center = degrees(atan2(z, sqrt(x * x + y * y)))
    lng_center = degrees(atan2(y, x))
    return lat_center, lng_center

def read_data():
    def gowalla():
        # gowalla
        df = pd.read_table("../raw_data/gowalla/Gowalla_totalCheckins.txt", names=["user", "dt", "lat", "lng", "poi"])
        df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(None)
        df = df.sort_values(["user", "dt"]).reset_index(drop=True)
        poi_df = df[["poi", "lat", "lng"]].drop_duplicates("poi").reset_index(drop=True)
        checkin_df = df[["user", "dt", "poi"]]
        poi_df.to_csv("./prepared_data/gowalla_poi.csv", index=False)
        checkin_df.to_csv("./prepared_data/gowalla_checkin.csv", index=False)
    
    def brightkite():
        # brightkite
        df = pd.read_table("../raw_data/brightkite/Brightkite_totalCheckins.txt", names=["user", "dt", "lat", "lng", "poi"])
        df["dt"] = pd.to_datetime(df["dt"]).dt.tz_localize(None)
        df = df.sort_values(["user", "dt"]).reset_index(drop=True)
        poi_df = df[["poi", "lat", "lng"]].drop_duplicates("poi").reset_index(drop=True)
        checkin_df = df[["user", "dt", "poi"]]
        poi_df.to_csv("./prepared_data/brightkite_poi.csv", index=False)
        checkin_df.to_csv("./prepared_data/brightkite_checkin.csv", index=False)

    def foursquare():
        # foursquare_global_scale
        df = pd.read_table("../raw_data/foursquare_global_scale/dataset_TIST2015_Checkins.txt", names=["user", "poi", "dt", "offset"])
        data_new = []
        for user, poi, dt, dt_offset in tqdm(df.values):
            try:
                local_dt_str = str((datetime.datetime.strptime(dt, "%a %b %d %H:%M:%S %z %Y") + datetime.timedelta(minutes=dt_offset)).replace(tzinfo=None))
                data_new.append([user, poi, local_dt_str])
            except Exception as e:
                print(e)
        df = pd.DataFrame(data_new, columns=["user", "poi", "dt"])
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.sort_values(["user", "dt"]).reset_index(drop=True)
        checkin_df = df[["user", "dt", "poi"]]
        checkin_df.to_csv("./prepared_data/foursquare_global_scale_checkin.csv", index=False)

        poi_list = pd.unique(checkin_df["poi"]).tolist()
        poi_df = pd.read_table("../raw_data/foursquare_global_scale/dataset_TIST2015_POIs.txt", names=["poi", "lat", "lng", "category", "country"])
        poi_df = poi_df[["poi", "lat", "lng"]]
        poi_df = poi_df[poi_df["poi"].isin(poi_list)].drop_duplicates("poi").reset_index(drop=True)
        poi_df.to_csv("./prepared_data/foursquare_global_scale_poi.csv", index=False)

        # foursquare_global_scale_social
        df = pd.read_table("../raw_data/foursquare_global_scale_social/dataset_WWW_Checkins_anonymized.txt", names=["user", "poi", "dt", "offset"])
        data_new = []
        for user, poi, dt, dt_offset in tqdm(df.values):
            try:
                local_dt_str = str((datetime.datetime.strptime(dt, "%a %b %d %H:%M:%S %z %Y") + datetime.timedelta(minutes=dt_offset)).replace(tzinfo=None))
                data_new.append([user, poi, local_dt_str])
            except Exception as e:
                print(e)
        df = pd.DataFrame(data_new, columns=["user", "poi", "dt"])
        df["dt"] = pd.to_datetime(df["dt"])
        df = df.sort_values(["user", "dt"]).reset_index(drop=True)
        checkin_df = df[["user", "dt", "poi"]]
        checkin_df.to_csv("./prepared_data/foursquare_global_scale_social_checkin.csv", index=False)

        poi_list = pd.unique(checkin_df["poi"]).tolist()
        poi_df = pd.read_table("../raw_data/foursquare_global_scale_social/raw_POIs.txt", names=["poi", "lat", "lng", "category", "country"])
        poi_df = poi_df[["poi", "lat", "lng"]]
        poi_df = poi_df[poi_df["poi"].isin(poi_list)].drop_duplicates("poi").reset_index(drop=True)
        poi_df.to_csv("./prepared_data/foursquare_global_scale_social_poi.csv", index=False)

        # foursquare_nyc_tokyo
        for name in ["NYC", "TKY"]:
            df = pd.read_table(f"../raw_data/foursquare_nyc_tky/dataset_TSMC2014_{name}.txt", names=["user", "poi", "cat_id", "cat_name", "lat", "lng", "offset", "dt"], encoding="ISO-8859-1")
            data_new = []
            for user, poi, cat_id, cat_name, lat, lng, dt_offset, dt in tqdm(df.values):
                try:
                    local_dt_str = str((datetime.datetime.strptime(dt, "%a %b %d %H:%M:%S %z %Y") + datetime.timedelta(minutes=dt_offset)).replace(tzinfo=None))
                    data_new.append([user, poi, local_dt_str, lat, lng])
                except Exception as e:
                    print(e)
            df = pd.DataFrame(data_new, columns=["user", "poi", "dt", "lat", "lng"])
            df["dt"] = pd.to_datetime(df["dt"])
            df = df.sort_values(["user", "dt"]).reset_index(drop=True)
            poi_df = df[["poi", "lat", "lng"]].drop_duplicates("poi").reset_index(drop=True)
            checkin_df = df[["user", "dt", "poi"]]
            poi_df.to_csv(f"./prepared_data/foursquare_{name}_poi.csv", index=False)
            checkin_df.to_csv(f"./prepared_data/foursquare_{name}_checkin.csv", index=False)

def pretrain_align():
    dataset_name_list = ["foursquare_NYC", "foursquare_global_scale", "foursquare_global_scale_social"]
    foursquare_checkin_df = None
    foursquare_poi_df = None
    for dataset_name in tqdm(dataset_name_list):
        poi_df = pd.read_csv(f"./prepared_data/{dataset_name}_poi.csv", header=0)
        if foursquare_poi_df is None:
            foursquare_poi_df = poi_df
        else:
            foursquare_poi_df = pd.concat([foursquare_poi_df, poi_df], ignore_index=True)

        checkin_df = pd.read_csv(f"./prepared_data/{dataset_name}_checkin.csv", header=0)
        checkin_df["user"] = checkin_df["user"].astype(str) + dataset_name
        if foursquare_checkin_df is None:
            foursquare_checkin_df = checkin_df
        else:
            foursquare_checkin_df = pd.concat([foursquare_checkin_df, checkin_df], ignore_index=True)
    
    foursquare_checkin_df["dt"] = pd.to_datetime(foursquare_checkin_df["dt"])
    foursquare_poi_df = foursquare_poi_df[(foursquare_poi_df["lat"].between(-90, 90, inclusive="neither")) & (foursquare_poi_df["lng"].between(-180, 180, inclusive="neither")) & (foursquare_poi_df["lat"] != 0) & (foursquare_poi_df["lng"] != 0)]

    is_usa_list = foursquare_poi_df.parallel_apply(is_usa, axis=1).tolist()
    foursquare_poi_df = foursquare_poi_df[is_usa_list]
    foursquare_poi_df = foursquare_poi_df.drop_duplicates("poi")
    poi_checkin_count = foursquare_checkin_df["poi"].value_counts()
    foursquare_poi_df["count"] = poi_checkin_count.loc[foursquare_poi_df["poi"]].tolist()

    user_checkin_count = set(foursquare_checkin_df["user"].value_counts()[foursquare_checkin_df["user"].value_counts() > window].index)
    foursquare_checkin_df = foursquare_checkin_df[foursquare_checkin_df["user"].isin(user_checkin_count)]

    foursquare_poi_df = foursquare_poi_df.reset_index(drop=True)

    all_poi = set(foursquare_checkin_df["poi"]) & set(foursquare_poi_df["poi"])
    foursquare_poi_df = foursquare_poi_df[foursquare_poi_df["poi"].isin(all_poi)].reset_index(drop=True)
    foursquare_checkin_df = foursquare_checkin_df[foursquare_checkin_df["poi"].isin(all_poi)].reset_index(drop=True)

    foursquare_poi_df["poi_index"] = foursquare_poi_df.index.to_list()
    
    user2index_df = pd.DataFrame([{"user": user, "user_index": index} for index, user in enumerate(foursquare_checkin_df["user"].drop_duplicates())]).set_index("user")
    poi2index_df = foursquare_poi_df[["poi", "poi_index"]].set_index("poi")
    foursquare_checkin_df["user"] = user2index_df.loc[foursquare_checkin_df["user"]]["user_index"].values
    foursquare_checkin_df["poi"] = poi2index_df.loc[foursquare_checkin_df["poi"]]["poi_index"].values
    foursquare_checkin_df = foursquare_checkin_df.sort_values(["user", "dt"])

    np.save(f"./align_data/pretrain/foursquare_poi_df.npy", foursquare_poi_df.to_dict("list"))
    np.save(f"./align_data/pretrain/foursquare_checkin_df.npy", foursquare_checkin_df.to_dict("list"))

    user_traj_dict = {}
    for index, data in tqdm(foursquare_checkin_df.iterrows(), total=len(foursquare_checkin_df)):
        user, dt, poi = data["user"], data["dt"], data["poi"]
        if user not in user_traj_dict:
            user_traj_dict[user] = {"poi": [], "dt": []}
        user_traj_dict[user]["poi"].append(poi)
        user_traj_dict[user]["dt"].append([dt.year, dt.month, dt.day, dt.day_of_week, dt.hour, dt.minute])
    np.save(f"./align_data/pretrain/foursquare_trajectory.npy", arr=user_traj_dict)

    np.save(f"./align_data/pretrain/foursquare_user_list.npy", arr=list(user_traj_dict.keys()))

    year_list = foursquare_checkin_df["dt"].dt.year.drop_duplicates().sort_values().tolist()
    np.save(f"./align_data/pretrain/foursquare_year_list.npy", arr=year_list)

def pretrain_cluster():
    foursquare_poi_df = pd.DataFrame(np.load(f"./align_data/pretrain/foursquare_poi_df.npy", allow_pickle=True).tolist())[["poi_index", "lat", "lng", "count"]]
    pretrain_ball_tree = BallTree(np.deg2rad(foursquare_poi_df[["lat", "lng"]].values), leaf_size=100, metric="haversine")
    
    path_name_list = [f"path{i+1}" for i in range(len(km_list))]
    group_by_list = [path_name_list[: i + 1] for i in range(len(km_list) - 1)]

    km_trans_list = [km / 6371.0088 for km in km_list]
    layer_list = [{-1: foursquare_poi_df["poi_index"].values.tolist()}]
    poi2path_dict = {poi:[] for poi in set(foursquare_poi_df["poi_index"])}

    for layer_index, km_trans in enumerate(km_trans_list[:-1]):
        allocated_set = set()
        layer_poi_dict = dict()
        for kernel_poi, poi_list in tqdm(layer_list[layer_index].items()):
            p_set = set(poi_list)
            classes = 0
            poi_df = foursquare_poi_df.iloc[poi_list].sort_values("count", ascending=False)
            for poi, lat, lng, count in poi_df.values:
                poi = int(poi)
                if poi in allocated_set:
                    continue
                query_index = pretrain_ball_tree.query_radius(np.deg2rad([lat, lng]).reshape(1, -1), r=km_trans)
                poi_list_ = list(dict.fromkeys([poi] + query_index[0].tolist()))
                poi_list_ = [p for p in poi_list_ if p in p_set and p not in allocated_set]
                for p in poi_list_:
                    poi2path_dict[p].append(classes)
                allocated_set.update(poi_list_) 
                layer_poi_dict[poi] = poi_list_
                classes += 1
        layer_list.append(layer_poi_dict)
    

    def fine(poi_list):
        sub_tree_geo = foursquare_poi_df.iloc[poi_list][["lat", "lng"]].to_numpy()
        clustering = DBSCAN(eps=km_list[-1]/6371.0088, min_samples=1, metric="haversine", n_jobs=1, algorithm="ball_tree").fit_predict(np.radians(sub_tree_geo))
        return clustering
    clustering_list = Parallel(n_jobs=-1)(delayed(fine)(poi_list) for poi_list in layer_list[-1].values())
    for poi_list, clustering in tqdm(zip(layer_list[-1].values(), clustering_list), total=len(clustering_list)):
        class2poi = dict()
        for p, class_ in zip(poi_list, clustering):
            poi2path_dict[p].append(class_)
            if class_ not in class2poi:
                class2poi[class_] = []
            class2poi[class_].append(p)
        for poi_list_ in class2poi.values():
            for index, p in enumerate(poi_list_):
                poi2path_dict[p].append(index)

    def freq(data):
        km_trans = 0.5 / 6371.0088
        poi, lat, lng = int(data["poi_index"]), data["lat"], data["lng"]
        query_index = pretrain_ball_tree.query_radius(np.deg2rad([lat, lng]).reshape(1, -1), r=km_trans)
        index_list = list(dict.fromkeys([poi] + query_index[0].tolist()))
        poi_df_ = foursquare_poi_df.iloc[index_list]
        freq_result = poi_df_.iloc[0]["count"] / poi_df_["count"].sum()
        return [poi, freq_result]

    freq_list = Parallel(n_jobs=-1)(delayed(freq)(data) for index, data in foursquare_poi_df.iterrows())
    poi2_freq = dict()
    for poi, freq_value in freq_list:
        poi2_freq[poi] = freq_value

    path_list = np.split(np.array([poi2path_dict[p] for p in foursquare_poi_df["poi_index"]]), indices_or_sections=len(km_list) + 1, axis=-1)
    path_name_list = []
    for index, path in enumerate(path_list):
        path_name = f"path{index+1}"
        foursquare_poi_df.insert(foursquare_poi_df.shape[1], path_name, path.squeeze(-1).tolist())
        path_name_list.append(path_name)
    foursquare_poi_df["freq"] = [poi2_freq[p] for p in foursquare_poi_df["poi_index"]]
    group_by_list = [path_name_list[: i + 1] for i in range(len(path_name_list) - 1)]

    constraint_dict = dict()
    for index, by in enumerate(group_by_list):
        for p_name, p_df in tqdm(foursquare_poi_df.groupby(by)):
            p_name = "_".join(list(map(str, p_name)))
            p_list = list(set(p_df[f"path{index+2}"]))
            constraint_dict[p_name] = p_list
    
    np.save(os.path.join(save_path, f"pretrain/foursquare_poi_tree_df.npy"), arr=foursquare_poi_df.to_dict("list"))
    np.save(os.path.join(save_path, f"pretrain/constraint_dict.npy"), arr=constraint_dict)

    foursquare_poi2path = {data[0]: list(data[1:]) for data in foursquare_poi_df[["poi_index"] + path_name_list].values}
    np.save(os.path.join(save_path, f"pretrain/foursquare_poi2path.npy"), arr=foursquare_poi2path)
    
    foursquare_checkin_df = pd.DataFrame(np.load(f"./align_data/pretrain/foursquare_checkin_df.npy", allow_pickle=True).tolist())
    user_count = len(set(foursquare_checkin_df["user"]))

    layer_count = (foursquare_poi_df[path_name_list].max() + 1).values.tolist()
    pretrain_param = {"user_count": user_count, "layer_count": layer_count}
    np.save(os.path.join(save_path, f"pretrain/pretrain_param.npy"), arr=pretrain_param)

    tree = {p: dict() for p in set(foursquare_poi_df[path_name_list[0]])}
    
    for index, by in enumerate(group_by_list):
        for p_name, p_df in tqdm(foursquare_poi_df.groupby(by)):
            temp = tree
            for p in p_name:
                temp = temp[p]
            temp.update({p: dict() for p in set(p_df[f"path{index+2}"])})
    np.save(os.path.join(save_path, f"pretrain/foursquare_tree_dict.npy"), arr=tree)


def generate_pretrain_data():
    if not os.path.exists(f"./pretrain/pretrain_dataset"): os.makedirs(f"./pretrain/pretrain_dataset")
    pretrain_traj_dict = np.load(f"./align_data/pretrain/foursquare_trajectory.npy", allow_pickle=True).tolist()
    pretrain_data = []
    for user, traj in tqdm(pretrain_traj_dict.items()):
        traj_list = traj["poi"]
        dt_list = traj["dt"]
        traj_split_list = [[traj_list[i: i + window], traj_list[i + window]] for i in range(0, len(traj_list) - window)]
        dt_split_list = [[dt_list[i: i + window], dt_list[i + window]] for i in range(0, len(dt_list) - window)]
        pretrain_data.extend([[user, traj_seq, traj_label] for (traj_seq, traj_label), (dt_seq, dt_label) in zip(traj_split_list, dt_split_list)])

    pretrain_env = lmdb.open(f"./pretrain/pretrain_dataset", map_size=1099511627776)
    pretrain_txn = pretrain_env.begin(write=True)
    for index, data in enumerate(tqdm(pretrain_data)):
        data = pickle.dumps(data)
        pretrain_txn.put(str(index).encode(), data)
    
    pretrain_txn.commit()
    pretrain_env.close()



def finetune_traj():
    poi_df = pd.read_csv(f"./prepared_data/{dataset_name}_poi.csv", header=0)
    checkin_df = pd.read_csv(f"./prepared_data/{dataset_name}_checkin.csv", header=0)

    poi_df = poi_df[(poi_df["lat"].between(-90, 90, inclusive="neither")) & (poi_df["lng"].between(-180, 180, inclusive="neither")) & (poi_df["lat"] != 0) & (poi_df["lng"] != 0)]
    is_usa_list = poi_df.parallel_apply(is_usa, axis=1).tolist()
    poi_df = poi_df[is_usa_list]

    poi_checkin_count = checkin_df["poi"].value_counts()
    poi_df["count"] = poi_checkin_count.loc[poi_df["poi"]].tolist()

    user_checkin_count = set(checkin_df["user"].value_counts()[checkin_df["user"].value_counts() > window].index)
    checkin_df = checkin_df[checkin_df["user"].isin(user_checkin_count)]
    poi_checkin_count = set(checkin_df["poi"].value_counts()[checkin_df["poi"].value_counts() > window].index)
    checkin_df = checkin_df[(checkin_df["poi"].isin(poi_checkin_count))]

    poi_df = poi_df.reset_index(drop=True)

    all_poi = set(checkin_df["poi"]) & set(poi_df["poi"])
    poi_df = poi_df[poi_df["poi"].isin(all_poi)].reset_index(drop=True)
    poi_df["poi_index"] = poi_df.index.to_list()
    checkin_df = checkin_df[checkin_df["poi"].isin(all_poi)].reset_index(drop=True)

    user2index_df = pd.DataFrame([{"user": user, "user_index": index} for index, user in enumerate(checkin_df["user"].drop_duplicates())]).set_index("user")
    poi2index_df = poi_df[["poi", "poi_index"]].set_index("poi")
    
    checkin_df["poi"] = poi2index_df.loc[checkin_df["poi"]]["poi_index"].values
    checkin_df["user"] = user2index_df.loc[checkin_df["user"]]["user_index"].values
    checkin_df["dt"] = pd.to_datetime(checkin_df["dt"])
    checkin_df = checkin_df.sort_values(["user", "dt"])
    np.save(f"./align_data/finetune/{dataset_name}_poi_df.npy", arr=poi_df.to_dict("list"))
    np.save(f"./align_data/finetune/{dataset_name}_checkin_df.npy", arr=checkin_df.to_dict("list"))

    year_list = checkin_df["dt"].dt.year.drop_duplicates().sort_values().tolist()
    np.save(f"./align_data/finetune/{dataset_name}_year_list.npy", arr=year_list)

    user_traj_dict = {}
    for user, dt, poi in tqdm(checkin_df.values):
        if user not in user_traj_dict:
            user_traj_dict[user] = {"poi": [], "dt": [], "dt_raw": []}
        user_traj_dict[user]["poi"].append(poi)
        user_traj_dict[user]["dt"].append([dt.year, dt.month, dt.day, dt.day_of_week, dt.hour, dt.minute])
        user_traj_dict[user]["dt_raw"].append(dt)
    np.save(f"./align_data/finetune/{dataset_name}_trajectory.npy", arr=user_traj_dict)


def generate_cold_start_dataset():
    random.seed(0)
    checkin_df = pd.DataFrame(np.load(f"./align_data/finetune/{dataset_name}_checkin_df.npy", allow_pickle=True).tolist())
    poi_freq = dict(checkin_df["poi"].value_counts())
    np.save(f"./cold_dataset/{dataset_name}/poi_freq.npy", arr=poi_freq)

    checkin_df = checkin_df.sort_values("dt")
    train_len = int(len(checkin_df) * 0.8)
    train_df, test_df = checkin_df[:train_len], checkin_df[train_len:]
    
    user_traj_dict = {}
    for index, (user, dt, poi) in enumerate(tqdm(checkin_df.values)):
        if user not in user_traj_dict:
            user_traj_dict[user] = {"train_poi": [], "train_dt": [], "test_poi": [], "test_dt": []}
        if index < train_len:
            user_traj_dict[user]["train_poi"].append(poi)
            user_traj_dict[user]["train_dt"].append([dt.year, dt.month, dt.day, dt.day_of_week, dt.hour, dt.minute])
        else:
            user_traj_dict[user]["test_poi"].append(poi)
            user_traj_dict[user]["test_dt"].append([dt.year, dt.month, dt.day, dt.day_of_week, dt.hour, dt.minute])

    train_set, test_set = set(train_df["poi"]), set(test_df["poi"])
    intersection = train_set & test_set
    train_value_counts = train_df[train_df["poi"].isin(intersection)]["poi"].value_counts()

    
    for count in cold_count_list:
        index_set = set(train_value_counts[train_value_counts == count].index)
        np.save(f"./cold_dataset/{dataset_name}/cold_list_{count}.npy", arr=list(index_set))
        train_list, test_list = [], []
        for user, traj_data in tqdm(user_traj_dict.items()):
            train_poi_seq, train_dt_seq = traj_data["train_poi"], traj_data["train_dt"]
            test_poi_seq, test_dt_seq = train_poi_seq[-window:] + traj_data["test_poi"], train_dt_seq[-window:] + traj_data["test_dt"]

            train_poi_seg_list = [[train_poi_seq[i:i+window], train_poi_seq[i+window]] for i in range(0, len(train_poi_seq) - window, 1)]
            train_dt_seg_list = [[train_dt_seq[i:i+window], train_dt_seq[i+window]] for i in range(0, len(train_dt_seq) - window, 1)]
            train_list.extend([[user, poi_seq, dt_seq, poi_label, dt_label] for (poi_seq, poi_label), (dt_seq, dt_label) in zip(train_poi_seg_list, train_dt_seg_list)])

            test_poi_seg_list = [[test_poi_seq[i:i+window], test_poi_seq[i+window]] for i in range(0, len(test_poi_seq) - window, 1)]
            test_dt_seg_list = [[test_dt_seq[i:i+window], test_dt_seq[i+window]] for i in range(0, len(test_dt_seq) - window, 1)]
            test_list.extend([[user, poi_seq, dt_seq, poi_label, dt_label] for (poi_seq, poi_label), (dt_seq, dt_label) in zip(test_poi_seg_list, test_dt_seg_list) if poi_label in index_set])
        

        np.save(f"./cold_dataset/{dataset_name}/cold_dataset_{count}.npy", arr={"train": train_list, "test": test_list})

def finetune_align():
    def freq(data):
        km = 0.5 / 6371.0088
        finetune_poi, finetune_lat, finetune_lng = int(data["poi_index"]), data["lat"], data["lng"]
        finetune_radius_index = finetune_tree.query_radius(np.deg2rad([finetune_lat, finetune_lng]).reshape(1, -1), km)
        finetune_radius_index_list = list(dict.fromkeys([finetune_poi] + finetune_radius_index[0].tolist()))
        poi_df_ = finetune_poi_df.iloc[finetune_radius_index_list]
        if poi_df_["count"].sum() != 0:
            freq = poi_df_.iloc[0]["count"] / poi_df_["count"].sum()
        else:
            freq = 0
        return [finetune_poi, freq]
    

    finetune_dir = os.path.join(save_path, "finetune", dataset_name)
    if not os.path.exists(finetune_dir): os.makedirs(finetune_dir)

    poi_freq_dict = np.load(f"./cold_dataset/{dataset_name}/poi_freq.npy", allow_pickle=True).tolist()

    pretrain_param = np.load(os.path.join(save_path, f"pretrain/pretrain_param.npy"), allow_pickle=True).tolist()
    layer_count = pretrain_param["layer_count"]

    pretrain_tree_df = pd.DataFrame(np.load(os.path.join(save_path, f"pretrain/foursquare_poi_tree_df.npy"), allow_pickle=True).tolist())
    
    poi_df_save_path = os.path.join(finetune_dir, f"{dataset_name}_poi_df.npy")
    finetune_poi_df = pd.DataFrame(np.load(f"./align_data/finetune/{dataset_name}_poi_df.npy", allow_pickle=True).tolist())
    finetune_poi_df["count"] = [poi_freq_dict[poi] for poi in finetune_poi_df["poi_index"]]
    np.save(poi_df_save_path, arr=finetune_poi_df.to_dict("list"))

    path_name_list = [f"path{i+1}" for i in range(len(layer_count))]
    finetune_poi_df_sort = finetune_poi_df.sort_values("count", ascending=False)
    finetune_tree = BallTree(np.deg2rad(finetune_poi_df[["lat", "lng"]].values), leaf_size=100, metric="haversine")

    finetune_poi2freq_dict = dict()
    
    for index, data in tqdm(finetune_poi_df_sort.iterrows(), total=len(finetune_poi_df_sort)):
        km = 0.5 / 6371.0088
        finetune_poi, finetune_lat, finetune_lng = int(data["poi_index"]), data["lat"], data["lng"]
        finetune_radius_index = finetune_tree.query_radius(np.deg2rad([finetune_lat, finetune_lng]).reshape(1, -1), km)
        finetune_radius_index_list = list(dict.fromkeys([finetune_poi] + finetune_radius_index[0].tolist()))
        poi_df_ = finetune_poi_df.iloc[finetune_radius_index_list]
        if poi_df_["count"].sum() != 0:
            freq_ = poi_df_.iloc[0]["count"] / poi_df_["count"].sum()
        else:
            freq_ = 0
        finetune_poi2freq_dict[finetune_poi] = freq_

    prefix2data, poi2prefix = dict(), dict()
    for group_name, group_df in tqdm(pretrain_tree_df.groupby(path_name_list[:-1])):
        prefix2data[group_name] = group_df[["poi_index", "freq"]]
        for poi in group_df["poi_index"]:
            poi2prefix[poi] = group_name

    match_dict = dict()
    temp_df = pretrain_tree_df.copy()
    search_count = 3000
    pretrain_tree = BallTree(np.deg2rad(temp_df[["lat", "lng"]].values), leaf_size=100, metric="haversine")
    for index, data in enumerate(tqdm(finetune_poi_df.itertuples(), total=len(finetune_poi_df))):
        finetune_poi, finetune_lat, finetune_lng = int(data.poi_index), data.lat, data.lng
        pretrain_match_list = pretrain_tree.query(np.deg2rad([finetune_lat, finetune_lng]).reshape(1, -1), k=min(search_count, len(temp_df)), return_distance=False)[0]
        find_flag = False
        for match_index in pretrain_match_list:
            if find_flag: break
            pretrain_poi = int(temp_df.iloc[match_index]["poi_index"])
            if pretrain_poi in match_dict: continue
            prefix = poi2prefix[pretrain_poi]
            prefix_df = prefix2data[prefix]
            if len(prefix_df) > 0:
                sort_index_list = np.argsort(np.abs(finetune_poi2freq_dict[finetune_poi] - prefix_df["freq"].values))
                for sort_index in sort_index_list:
                    match_poi = int(prefix_df.iloc[sort_index]["poi_index"])
                    if match_poi not in match_dict:
                        match_dict[match_poi] = finetune_poi
                        prefix2data[prefix] = prefix_df.drop(index=match_poi)
                        find_flag = True
                        break
            else:
                del prefix2data[prefix]

        if index % search_count == 0:
            temp_df = pretrain_tree_df.drop(set(match_dict))
            pretrain_tree = BallTree(np.deg2rad(temp_df[["lat", "lng"]].values), leaf_size=100, metric="haversine")

    np.save(os.path.join(save_path, f"finetune/{dataset_name}/poi_match_dict.npy"), arr=match_dict)

def align_user():
    pretrain_trajectory = np.load("./align_data/pretrain/foursquare_trajectory.npy", allow_pickle=True).tolist()
    finetune_trajectory = np.load(f"./align_data/finetune/{dataset_name}_trajectory.npy", allow_pickle=True).tolist()
    pretrain2finetune_dict = np.load(f"./align_data/{dir_name}/finetune/{dataset_name}/poi_match_dict.npy", allow_pickle=True).tolist()
    finetune2pretrain_dict = {value: key for key, value in pretrain2finetune_dict.items()}
    pretrain_id_set, finetune_id_set = set(pretrain2finetune_dict.keys()), set(pretrain2finetune_dict.values())
    poi_intersection = pretrain_id_set & finetune_id_set
    finetune_poi_traj = {user: Counter([finetune2pretrain_dict[poi] for poi in data["poi"]]) for user, data in tqdm(finetune_trajectory.items())}

    pretrain_poi_traj = {user: Counter([poi if poi in pretrain_id_set else -1 for poi in data["poi"]]) for user, data in tqdm(pretrain_trajectory.items())}

    finetune_poi_traj_list = [[user, traj] for user, traj in finetune_poi_traj.items()]
    list_size = len(finetune_poi_traj_list)
    process = 16
    split_len = list_size // (process)
    split_list = [finetune_poi_traj_list[i:i+split_len] for i in range(0, list_size, split_len)]
    pretrain_poi_traj_list = [[user, traj] for user, traj in pretrain_poi_traj.items()]

    def calculate_score_multi(finetune_poi_traj_list):
        result = []
        for finetune_user, finetune_traj_counter in tqdm(finetune_poi_traj_list):
            finetune_traj_set = set(finetune_traj_counter.keys())
            finetune_traj_len = sum(finetune_traj_counter.values())
            score_dict = dict()
            for foursquare_user, foursquare_traj_counter in pretrain_poi_traj_list:
                foursquare_traj_set = set(foursquare_traj_counter.keys())
                if not finetune_traj_set.isdisjoint(foursquare_traj_set):
                    foursquare_traj_len = sum(foursquare_traj_counter.values())
                    intersection_count = sum((finetune_traj_counter & foursquare_traj_counter).values())
                    denominator = finetune_traj_len + foursquare_traj_len - intersection_count
                    if denominator == 0:
                        score = 1
                    else:
                        score = intersection_count / denominator
                else:
                    score = 0
                score_dict[foursquare_user] = score
            score_dict = dict(sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:1000])
            result.append([finetune_user, score_dict])
        return result

    result = Parallel(n_jobs=process, backend="loky")(delayed(calculate_score_multi)(split) for split in tqdm(split_list))
    np.save(f"./align_data/finetune/{dataset_name}_user_align_tanimoto.npy", arr=np.array(result, dtype=object))

def match_user():
    user_result = np.load(f"./align_data/finetune/{dataset_name}_user_align_tanimoto.npy", allow_pickle=True).tolist()
    user2match_dict = {}
    full_zero_user = []
    for process_result in user_result:
        for user, match_dict in process_result:
            if max(match_dict.values()) == 0:
                full_zero_user.append(user)
            else:
                match_score_dict = {match: score for match, score in match_dict.items() if score != 0}
                match_score_dict = dict(sorted(match_score_dict.items(), key=lambda x: x[1], reverse=True))
                user2match_dict.update({user: match_score_dict})

    user_len = len(user2match_dict)
    
    match2user_dict = dict()
    for user, score_dict in user2match_dict.items():
        for match, score in score_dict.items():
            if match not in match2user_dict:
                match2user_dict[match] = dict()
            match2user_dict[match][user] = score
    match2user_dict = {key: dict(sorted(value.items(), key=lambda x: x[1], reverse=True)) for key, value in match2user_dict.items()}

    user_align = dict()
    match_pool = set()

    for match, user_score_dict in tqdm(match2user_dict.items()):
        if match in match_pool:
            continue
        best_user = next(iter(user_score_dict))
        best_score = user_score_dict[best_user]
        if best_user in user_align and match in match_pool:
            continue
        elif best_user not in user_align and match not in match_pool:
            user_align[best_user] = [match, best_score]
            match_pool.add(match)
            continue
        elif best_user in user_align and match not in match_pool:
            for user, score in match2user_dict[match].items():
                if user not in user_align:
                    user_align[user] = [match, score]
                    match_pool.add(match)
                    break
            continue
        elif best_user not in user_align and match in match_pool:
            for match_, match_score_ in user2match_dict[best_user].items():
                if match_ not in match_pool:
                    user_align[best_user] = [match_, match_score_]
                    match_pool.add(match_)
                    break


    unallocate_user_set = set(user2match_dict.keys()) - set(user_align.keys())
    match2user = {match: [user, score] for user, (match, score) in user_align.items()}
    for user in tqdm(unallocate_user_set):
        find_flag = False
        for match, match_score in user2match_dict[user].items():
            if find_flag == True:
                break
            current_user, current_score = match2user[match]

            for match_, match_score_ in user2match_dict[current_user].items():
                if match_score_ < current_score:
                    break
                elif match_score_ >= current_score:
                    if match_ not in match_pool:

                        user_align[user] = [match, current_score]
                        user_align[current_user] = [match_, match_score_]
                        match_pool.add(match)
                        match_pool.add(match_)

                        match2user[match_] = [current_user, match_score]
                        match2user[match] = [user, current_score]
                        find_flag = True
                        break
        
        
    unallocate_user_set = list((set(user2match_dict.keys()) - set(user_align.keys())) | set(full_zero_user))
    unallocate_match_set = list(set(match2user_dict.keys()) - match_pool)
    for index, unallocate_user in enumerate(unallocate_user_set):
        user_align[unallocate_user] = [unallocate_match_set[index], -1]

    
    zero_user_count = len(full_zero_user)
    allocate_count = len(set([value[0] for key, value in user_align.items()]))
    print("user_align", len(user_align), "total", user_len, "allocate_count", allocate_count, allocate_count + zero_user_count)
    user_align = {key: value[0] for key, value in user_align.items()}
    np.save(f"./align_data/finetune/{dataset_name}_user_align.npy", arr=user_align)

def generate_finetune_data():
    pretrain_data_path = "./pretrain/pretrain_dataset"
    pretrain_env = lmdb.open(pretrain_data_path, readonly=True)
    pretrain_txn = pretrain_env.begin()
    pretrain_index_list = [str(i) for i in range(pretrain_txn.stat()["entries"])]
    poi_trans_dict = np.load(f"./align_data/{dir_name}/finetune/{dataset_name}/poi_match_dict.npy", allow_pickle=True).tolist()
    user_trans_dict = np.load(f"./align_data/finetune/{dataset_name}_user_align.npy", allow_pickle=True).tolist()
    user_trans_dict = {value: key for key, value in user_trans_dict.items()}

    for cold_count in cold_count_list:
        cold_poi_set = set(np.load(f"./cold_dataset/{dataset_name}/cold_list_{cold_count}.npy", allow_pickle=True).tolist())

        dataset = np.load(f"./cold_dataset/{dataset_name}/cold_dataset_{cold_count}.npy", allow_pickle=True).tolist()
        train_dataset, test_dataset = dataset["train"], dataset["test"]

        cold_train = [[user_id, poi_list, poi_label, "c"] for user_id, poi_list, dt_list, poi_label, dt_label in train_dataset]
        cold_test = [[user_id, poi_list, poi_label, "c"] for user_id, poi_list, dt_list, poi_label, dt_label in test_dataset]

        pretrain_add_data = []
        for dataset_index in tqdm(pretrain_index_list):
            user, traj_seq, traj_label = pickle.loads(pretrain_txn.get(dataset_index.encode()))

            if poi_trans_dict.get(traj_label, -1) in cold_poi_set:
                pretrain_add_data.append([user, traj_seq, traj_label, "p"])

        np.save(f"./cold_dataset/{dataset_name}/cold_dataset_{cold_count}_finetune.npy", arr={"train": cold_train, "test": cold_test, "pretrain": pretrain_add_data})


def generate_finetune_constraint_dict():
    for cold_count in cold_count_list:
        path_name_list = [f"path{i+1}" for i in range(len(km_list)+1)]
        group_by_list = [path_name_list[: i + 1] for i in range(len(path_name_list) - 1)]
        dataset = np.load(f"./cold_dataset/{dataset_name}/cold_dataset_{cold_count}_finetune.npy", allow_pickle=True).tolist()
        pretrain_dataset = dataset["pretrain"]
        all_pretrain_label = set([data[2] for data in pretrain_dataset])
        poi_trans_dict = np.load(f"./align_data/{dir_name}/finetune/{dataset_name}/poi_match_dict.npy", allow_pickle=True).tolist()
        pretrain_poi2path = np.load(f"./align_data/{dir_name}/pretrain/foursquare_poi2path.npy", allow_pickle=True).tolist()
        finetune_path_df = [[finetune_poi] + pretrain_poi2path[pretrain_poi] for pretrain_poi, finetune_poi in poi_trans_dict.items()]
        finetune_path_df.extend([[-1] + pretrain_poi2path[poi] for poi in all_pretrain_label])
        finetune_path_df = pd.DataFrame(finetune_path_df, columns=["poi"] + path_name_list)

        constraint_dict = {"-1": list(set(finetune_path_df["path1"]))}
        for index, by in enumerate(group_by_list):
            for p_name, p_df in tqdm(finetune_path_df.groupby(by)):
                p_name = "_".join(list(map(str, p_name)))
                p_list = p_df[f"path{index+2}"].drop_duplicates().tolist()
                constraint_dict[p_name] = p_list
        
        constraint_path = f"./align_data/{dir_name}/finetune/{dataset_name}/{cold_count}_constraint_dict.npy"
        np.save(constraint_path, arr=constraint_dict)


if __name__ == '__main__':
    read_data()
    pretrain_align()
    pretrain_cluster()
    generate_pretrain_data()
    finetune_traj()
    generate_cold_start_dataset()
    finetune_align()
    align_user()
    match_user()
    generate_finetune_data()
    generate_finetune_constraint_dict()
