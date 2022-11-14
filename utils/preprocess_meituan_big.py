import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict


def preprocess(train_data_name, val_data_name, test_data_name):
    u_list, i_list, ts_list, lat_list, lon_list, label_list = [], [], [], [], [], []
    feat_l = []
    idx_list = []

    u2i_map_list = defaultdict(list)
    i2u_map_list = defaultdict(list)

    u_index_list = []
    i_index_list = []

    u_map = {}
    i_map = {}

    u_ind = 0
    i_ind = 0

    user_count = defaultdict(int)
    item_count = defaultdict(int)

    with open(train_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            click = int(e[2])
            if click == 0:
                continue
            user_count[u] += 1
            item_count[i] += 1
        
    with open(val_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            click = int(e[2])
            if click == 0:
                continue
            user_count[u] += 1
            item_count[i] += 1
    
    with open(test_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            click = int(e[2])
            if click == 0:
                continue
            user_count[u] += 1
            item_count[i] += 1

    num_interaction_train = 0
    num_interaction_val = 0
    num_interaction_test = 0

    with open(train_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            lat = float(e[3])
            lon = float(e[4])
            click = int(e[2])
            if click == 0:
                continue
            if user_count[u] < 5 or item_count[i] < 5:
                continue

            num_interaction_train += 1

            if u not in u_map:
                u_map[u] = u_ind
                u_ind += 1
            if i not in i_map:
                i_map[i] = i_ind
                i_ind += 1
            
            u = u_map[u]
            i = i_map[i]

            ts = float(e[5])
            label = float(1.)  # int(e[3])

            # feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            lat_list.append(lat)
            lon_list.append(lon)
    
    with open(val_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            lat = float(e[3])
            lon = float(e[4])
            click = int(e[2])
            if click == 0:
                continue
            if user_count[u] < 5 or item_count[i] < 5:
                continue

            num_interaction_val += 1

            if u not in u_map:
                u_map[u] = u_ind
                u_ind += 1
            if i not in i_map:
                i_map[i] = i_ind
                i_ind += 1
            
            u = u_map[u]
            i = i_map[i]

            ts = float(e[5])
            label = float(1.)  # int(e[3])

            # feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            lat_list.append(lat)
            lon_list.append(lon)

    with open(test_data_name) as f:
        s = next(f)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = e[0]
            i = e[1]
            lat = float(e[3])
            lon = float(e[4])
            click = int(e[2])
            if click == 0:
                continue
            if user_count[u] < 5 or item_count[i] < 5:
                continue

            num_interaction_test += 1

            if u not in u_map:
                u_map[u] = u_ind
                u_ind += 1
            if i not in i_map:
                i_map[i] = i_ind
                i_ind += 1
            
            u = u_map[u]
            i = i_map[i]

            ts = float(e[5])
            label = float(1.)  # int(e[3])

            # feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            lat_list.append(lat)
            lon_list.append(lon)
    
    num_interaction = num_interaction_train + num_interaction_val + num_interaction_test
    print('edge: ', num_interaction)
    print(float(num_interaction_train) / float(num_interaction))
    print(float(num_interaction_train + num_interaction_val) / float(num_interaction))

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
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        print(new_df.u.max())
        print(new_df.i.max())

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
    train_path = './data/{}/big_train.csv'.format(data_name, data_name)
    val_path = './data/{}/big_val.csv'.format(data_name, data_name)
    test_path = './data/{}/big_test.csv'.format(data_name, data_name)
    OUT_DF = './data/{}/ml_{}.csv'.format(data_name, data_name)
    OUT_FEAT = './data/{}/ml_{}.npy'.format(data_name, data_name)
    OUT_NODE_FEAT = './data/{}/ml_{}_node.npy'.format(data_name, data_name)

    df = preprocess(train_path, val_path, test_path)  # , feat
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
                    default='meituan_big')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)
