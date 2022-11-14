import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time
from collections import defaultdict


def time2stamp(dt):
    timeArray = time.strptime(dt, "%Y-%m-%dT%H:%M:%SZ")
    timestamp = time.mktime(timeArray)
    return timestamp


def preprocess(data_name):
    u_list, i_list, ts_list, lat_list, lon_list, label_list = [], [], [], [], [], []
    # feat_l = []
    idx_list = []

    u_map = {}
    i_map = {}
    u_ind = 0
    i_ind = 0
    u_count = defaultdict(int)
    i_count = defaultdict(int)

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split('\t')
            u = int(e[0])
            i = int(e[4])
            u_count[u] += 1
            i_count[i] += 1

    with open(data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split('\t')
            u = int(e[0])
            i = int(e[4])
            lat = float(e[2])
            lon = float(e[3])
            if u_count[u] < 10 or i_count[i] < 10:
                continue

            if u not in u_map:
                u_map[u] = u_ind               
                u_ind += 1
            if i not in i_map:
                i_map[i] = i_ind
                i_ind += 1
            
            u = u_map[u]
            i = i_map[i]

            ts = time2stamp(e[1])
            label = float(1)  # int(e[3])

            # feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            lat_list.append(lat)
            lon_list.append(lon)

            # feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'lat': lat_list,
                         'lon': lon_list,
                         'label': label_list,
                         'idx': idx_list}) #, np.array(feat_l)


def reindex(df, bipartite=True):
    new_df = df.copy()
    print(new_df.u.max())
    print(new_df.i.max())
    
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1
    else:
        new_df.u += 1
        new_df.i += 1
        new_df.idx += 1

    return new_df


def run(data_name, bipartite=True):
    Path(f"data/{data_name}").mkdir(parents=True, exist_ok=True)
    PATH = './data/{}/loc-gowalla_totalCheckins.txt'.format(data_name, data_name)
    OUT_DF = './data/{}/ml_{}.csv'.format(data_name, data_name)
    # OUT_FEAT = './data/{}/ml_{}.npy'.format(data_name, data_name)
    # OUT_NODE_FEAT = './data/{}/ml_{}_node.npy'.format(data_name, data_name)

    df = preprocess(PATH)  # , feat
    df.sort_values("ts", inplace=True)
    new_df = reindex(df, bipartite)

    # empty = np.zeros(feat.shape[1])[np.newaxis, :]
    # feat = np.vstack([empty, feat])

    # max_idx = max(new_df.u.max(), new_df.i.max())
    # rand_feat = np.zeros((max_idx + 1, 172))

    new_df.to_csv(OUT_DF)
    # np.save(OUT_FEAT, feat)
    # np.save(OUT_NODE_FEAT, rand_feat)


parser = argparse.ArgumentParser('Interface for LSTDR data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='gowalla')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)
