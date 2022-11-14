import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import multiprocessing
import torch
from tqdm import tqdm
from collections import defaultdict


Ks = [1, 5, 10, 20, 40, 50, 60, 70, 80, 90, 100, 200]

def eval_one_user(x):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks)), 'auc': 0., 'mrr': 0.}
    preds = np.transpose(x[0])

    pos_label = np.ones(1)

    num_preditems = x[1]

    uit = x[2]
    rec_items = x[3]

    num_neg_sample_items = x[4]
    num_candidate_items = x[5]

    labels = np.zeros(num_preditems)
    labels[0] = 1
    # scaler = MinMaxScaler()
    # posterior = np.transpose(scaler.fit_transform(np.transpose([preds])))[0]
    posterior = preds.copy()
    r = []
    rankeditems = np.argsort(-preds)[:max(Ks)]
    rank_scores = posterior[rankeditems[:max(Ks)]]
    for i in rankeditems:
        if i == 0:
            r.append(1)
        else:
            r.append(0)
    if num_neg_sample_items != -1:
        r = rank_corrected(np.array(r), num_preditems, num_candidate_items)

    precision, recall, ndcg, hit_ratio = [], [], [], []
    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, 1))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))
    auc_k = auc(ground_truth=labels, prediction=posterior)
    mrr_k = mrr(r)


    result['precision'] += precision
    result['recall'] += recall
    result['ndcg'] += ndcg
    result['hit_ratio'] += hit_ratio
    result['auc'] += auc_k
    result['mrr'] += mrr_k

    return (result, rankeditems[:max(Ks)], uit, rec_items, rank_scores)


def rank_corrected(r, m, n):
    pos_ranks = np.argwhere(r==1)[:,0]
    corrected_r = np.zeros_like(r)
    for each_sample_rank in list(pos_ranks):
        corrected_rank = int(np.floor(((n-1)*each_sample_rank)/m))
        if corrected_rank >= len(corrected_r) - 1:
            continue
        corrected_r[corrected_rank] = 1
    assert sum(corrected_r) <= 1
    return corrected_r


def eval_users(lstdr, src, dst, ts, lats, lons, eidx, train_src, train_dst, args):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
            'hit_ratio': np.zeros(len(Ks)), 'auc': 0., 'mrr': 0.}
    cores = multiprocessing.cpu_count() // 2
    userset = set(src)
    train_itemset = set(train_dst)
    # print(len(train_itemset))
    pos_edges = {}
    for u, i, t in zip(src, dst, ts):
        if i not in train_itemset:
            continue
        if u in pos_edges:
            pos_edges[u].add((i, t))
        else:
            pos_edges[u] = set([(i, t)])
    train_pos_edges = {}
    for u, i in zip(train_src, train_dst):
        if u in train_pos_edges:
            train_pos_edges[u].add(i)
        else:
            train_pos_edges[u] = set([i])

    pool = multiprocessing.Pool(cores)
    batch_users = 5

    preds_list = []
    preds_len_preditems = []
    preds_uit = []
    preds_rec_items = []
    preds_sampled_neg = []
    preds_num_candidates = []

    # test_outputs = []
    test_outputs = defaultdict(list)

    num_interactions = 0
    num_test_instances = 0
    num_user = {}
    with torch.no_grad():
        lstdr = lstdr.eval()
        batch_src_l = []
        batch_test_items = []
        batch_ts = []
        batch_lat, batch_lon = [], []
        batch_eidx = []
        #batch_len = []
        batch_i = 0
        last_lat = {}
        last_time = {}
        last_lon = {}
        start_lat = {}
        start_lon = {}
        for u, i, t, lat, lon, e in tqdm(zip(src, dst, ts, lats, lons, eidx), desc="Recommending "):
            num_test_instances += 1
            if u not in train_src or i not in train_itemset or u not in pos_edges:
                continue
            if u not in num_user:
                num_user[u] = 1
            num_interactions += 1
            batch_i += 1

            pos_items = [i]
            pos_ts = [t]
            if (u not in last_lat) or (u not in last_lon):
                last_lat[u] = lat
                last_lon[u] = lon
                start_lat[u] = lat
                start_lon[u] = lon
                last_time[u] = t

            pos_lat = [start_lat[u]] 
            pos_lon = [start_lon[u]]
            # if args.data == 'gowalla':
            #     pos_lat = [start_lat[u]] # lat # last_lat[u] # start_lat[u]
            #     pos_lon = [start_lon[u]] # lon # last_lon[u] # start_lon[u]
            # else:
            #     # pos_lat = [last_lat[u]] # lat # last_lat[u] # start_lat[u]
            #     # pos_lon = [last_lon[u]]
            #     pos_lat = [start_lat[u]] # lat # last_lat[u] # start_lat[u]
            #     pos_lon = [start_lon[u]] # lon # last_lon[u] # start_lon[u]
            last_lat[u] = lat
            last_lon[u] = lon
            last_time[u] = t

            pos_eidx = [e]
            src_l = [u for _ in range(len(pos_items))]
            pos_label = np.ones(len(pos_items))

            interacted_dst = train_pos_edges[u]

            neg_candidates = list(train_itemset - set(pos_items) - interacted_dst)
            # print("neg list length: ", len(neg_candidates))
            if args.negsampleeval == -1:
                neg_items = neg_candidates
            else:
                neg_items = list(np.random.choice(neg_candidates, size=args.negsampleeval, replace=False))
            #neg_items = list(train_itemset - set(pos_items))
            neg_ts = [t for _ in range(len(neg_items))]
            neg_lat = [lat for _ in range(len(neg_items))] # lat # start_lat[u]
            neg_lon = [lon for _ in range(len(neg_items))] # lon # start_lon[u]
            neg_eidx = [e for _ in range(len(neg_items))]
            neg_src_l = [u for _ in range(len(neg_items))]

            batch_src_l += src_l + neg_src_l
            batch_test_items += pos_items + neg_items
            batch_ts += pos_ts + neg_ts
            batch_lat += pos_lat + neg_lat
            batch_lon += pos_lon + neg_lon
            batch_eidx += pos_eidx + neg_eidx
            #batch_len.append(len(src_l+neg_src_l))

            test_items = np.array(batch_test_items)
            test_ts = np.array(batch_ts)
            test_lat = np.array(batch_lat)
            test_lon = np.array(batch_lon)
            test_eidx = np.array(batch_eidx)
            test_src_l = np.array(batch_src_l)

            pred_scores = lstdr(test_src_l, test_items, test_ts, test_lat, test_lon, test_eidx, n_neighbors=args.n_degree)  # lstdr forward
            preds = pred_scores.sigmoid().cpu().numpy()
            #start_ind = 0
            #for i_len in batch_len:
            preds_list.append(preds)
            preds_len_preditems.append(len(src_l+neg_src_l))
            preds_uit.append((u,i,t, last_lat[u], last_lon[u], last_time[u]))
            rec_items = []
            rec_items += pos_items + neg_items
            preds_rec_items.append(rec_items)
            preds_sampled_neg.append(args.negsampleeval)
            preds_num_candidates.append(len(pos_items+neg_candidates))
                #start_ind = i_len
            batch_src_l = []
            batch_test_items = []
            batch_ts = []
            batch_lat = []
            batch_lon = []
            batch_eidx = []
            #batch_len = []

            if len(preds_list) % batch_users == 0 or num_test_instances == len(ts):

                batchset_predictions = zip(preds_list, preds_len_preditems, preds_uit, preds_rec_items, preds_sampled_neg, preds_num_candidates)
                batch_preds = pool.map(eval_one_user, batchset_predictions)
                for oneresult in batch_preds:
                    re = oneresult[0]
                    result['precision'] += re['precision']
                    result['recall'] += re['recall']
                    result['ndcg'] += re['ndcg']
                    result['hit_ratio'] += re['hit_ratio']
                    result['auc'] += re['auc']
                    result['mrr'] += re['mrr']

                    uit = oneresult[2]
                    pred_rank_list = oneresult[1]
                    rec_items = oneresult[3]
                    pred_score_list = oneresult[4]

                    one_pred_result = {"u_pos_gd": int(uit[1]), "timestamp": float(uit[2]), "last_lat": uit[3], "last_lon": uit[4], "last_time": uit[5]}  # 
                    one_pred_result["predicted"] = [int(rec_items[int(rec_ind)]) for rec_ind in pred_rank_list]
                    one_pred_result["scores"] = list(pred_score_list)
                    # test_outputs.append(one_pred_result)
                    test_outputs[int(uit[0])].append(one_pred_result)


                preds_list = []
                preds_len_preditems = []
                preds_uit = []
                preds_rec_items = []
                preds_sampled_neg = []
                preds_num_candidates = []
                batch_src_l = []
                batch_test_items = []
                batch_ts = []
                batch_lat = []
                batch_lon = []
                #batch_len = []

    result['precision'] /= num_interactions
    result['recall'] /= num_interactions
    result['ndcg'] /= num_interactions
    result['hit_ratio'] /= num_interactions
    result['auc'] /= num_interactions
    result['mrr'] /= num_interactions
    print('num_interactions: ', num_interactions)

    return result, test_outputs


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in tqdm(range(num_test_batch), desc="Link prediction "):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            lats_batch = data.lats[s_idx:e_idx]
            lons_batch = data.lons[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

            size = len(sources_batch)
            # _, negative_samples = negative_edge_sampler.sample(size)
            negative_samples = negative_edge_sampler.sample_neg(sources_batch)

            pos_prob, neg_prob = model.compute_edge_probabilities_nosigmoid(sources_batch, destinations_batch,
                                                                            negative_samples, timestamps_batch, lats_batch, lons_batch,
                                                                            edge_idxs_batch, n_neighbors)

            pred_score = np.concatenate([(pos_prob.sigmoid()).cpu().numpy(), (neg_prob.sigmoid()).cpu().numpy()])
            true_label = np.concatenate([np.ones(size), np.zeros(size)])

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)  # val_ap, val_auc


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx: e_idx]
            destinations_batch = data.destinations[s_idx: e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx: e_idx]

            source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                         destinations_batch,
                                                                                         destinations_batch,
                                                                                         timestamps_batch,
                                                                                         edge_idxs_batch,
                                                                                         n_neighbors)
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc


def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def mrr(r):
    r = np.array(r)
    if np.sum(r) == 0:
        return 0.
    return np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]