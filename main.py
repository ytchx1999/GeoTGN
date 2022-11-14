from ast import arg
import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

# from evaluation.evaluation import eval_edge_prediction
from evaluation.evaluation import *
from model.tgn import LSTDR
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import random
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(2022)
# torch.manual_seed(2022)
# np.random.seed(2022)

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


### Argument and global variables
parser = argparse.ArgumentParser('LSTDR')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)', default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true', help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=["graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=["mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=["gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true', help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for each user')
parser.add_argument('--different_new_nodes', action='store_true', help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true', help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true', help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true', help='Whether to run the dyrep model')
parser.add_argument('--pos_encode', action='store_true', help='Whether to run the pos encoding')
parser.add_argument('--reg', type=float, default=0.1, help='regularization')
parser.add_argument('--negsampleeval', type=int, default=-1, help='number of negative sampling evaluation, -1 for all')
parser.add_argument('--start_percent', type=float, default=0.0, help='regularization')
parser.add_argument('--seed', type=int, default=2022, help='Seed for all')
parser.add_argument('--loss', type=str, default='bpr', help='Loss function')
parser.add_argument('--no_norm', action='store_true', help='Whether to use LayerNorm in MergeLayer')
parser.add_argument('--data_path', type=str, default="/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/chihuixuan/MT-git/local-tgn", help='path of data')  # "./"
parser.add_argument('--model_path', type=str, default="/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/chihuixuan/MT-git/local-tgn", help='path of model')   # "./"
parser.add_argument('--log_path', type=str, default="/home/hadoop-aipnlp/dolphinfs/hdd_pool/data/chihuixuan/MT-git/local-tgn", help='path of log')
parser.add_argument('--max_dist', type=float, default=100.0, help='regularization')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

set_seed(args.seed)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

# Path("./saved_models/").mkdir(parents=True, exist_ok=True)
# Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
# MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
# get_checkpoint_path = lambda \
#         epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path(f"{args.log_path}/log/").mkdir(parents=True, exist_ok=True)
# Path("scripts/log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('{}/log/{}.log'.format(args.log_path, str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
edge_features, full_data, train_data, val_data, test_data = get_data(DATA,
                                                                     different_new_nodes_between_val_and_test=args.different_new_nodes,
                                                                     randomize_features=args.randomize_features,
                                                                     start_percent=args.start_percent,
                                                                     data_path=args.data_path)

n_nodes = full_data.n_unique_nodes

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations, train_data.timestamps)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps, seed=2022)
# nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations,
#                                       seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, full_data.timestamps, seed=2022)
# nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources,
#                                        new_node_test_data.destinations,
#                                        seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)
# gpu_ids = [0, 1]

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

best_result_precision = []
best_result_recall = []
best_result_ndcg = []
best_result_hit = []
best_result_mrr = []

best_val_result_precision = []
best_val_result_recall = []
best_val_result_ndcg = []
best_val_result_hit = []
best_val_result_mrr = []

for i in range(args.n_runs):
    # results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    # Path("results/").mkdir(parents=True, exist_ok=True)
    print('-'*50, flush=True)
    print(f'Run {i}:', flush=True)

    set_seed(args.seed+i)

    # Initialize Model
    model = LSTDR(neighbor_finder=train_ngh_finder, n_nodes=(n_nodes + 1), node_dim=NODE_DIM,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep, pos=args.pos_encode, use_norm=(not args.no_norm), max_dist=args.max_dist)
    # criterion = torch.nn.BCELoss()
    # criterion = tgn.bpr_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = model.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    new_nodes_val_aps = []
    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    best_mrr = 0.
    best_result = None
    best_val_result = None
    best_outputs = None

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    for epoch in range(NUM_EPOCH):
        train_start_epoch = time.time()
        ### Training

        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            model.memory.__init_memory__()

        # Train using only training graph
        model.set_neighbor_finder(train_ngh_finder)
        # m_loss = []
        acc, m_loss = [], []
        ap, f1, auc = [], [], []

        logger.info('start {} epoch'.format(epoch))
        for k in tqdm(range(0, num_batch, args.backprop_every), desc=f"Run {i}, Epoch {epoch}"):
            loss = 0
            optimizer.zero_grad()
            model = model.train()

            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args.backprop_every):
                batch_idx = k + j

                if batch_idx >= num_batch:
                    continue

                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                    train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]
                lats_batch = train_data.lats[start_idx:end_idx]
                lons_batch = train_data.lons[start_idx:end_idx]

                size = len(sources_batch)
                # _, negatives_batch = train_rand_sampler.sample(size)
                negatives_batch = train_rand_sampler.sample_neg(sources_batch)

                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)

                model = model.train()
                # pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, negatives_batch,
                #                                                     timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
                pos_prob, neg_prob = model.compute_edge_probabilities_nosigmoid(sources_batch, destinations_batch, negatives_batch,
                                                                              timestamps_batch, lats_batch, lons_batch, edge_idxs_batch, NUM_NEIGHBORS)

                # loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)
                if args.loss == 'bpr':
                    loss += model.bpr_loss(pos_prob, neg_prob)
                    l2_reg = 0
                    for name, p in model.named_parameters():
                        if "node_raw_features" in name:
                            l2_reg += p.norm(2)
                    loss += (args.reg * l2_reg)
                elif args.loss == 'bce':
                    criterion = torch.nn.BCELoss()
                    loss += criterion(pos_prob.sigmoid().squeeze(), pos_label) + criterion(neg_prob.sigmoid().squeeze(), neg_label)

                # weight = None
                # for name, p in tgn.named_parameters():
                #     if "node_raw_features" in name:
                #         weight = p.norm(2)
                #         break
                # loss += tgn.weighted_hinge_auc_loss(pos_prob, neg_prob, num_neg=1, weight=weight)

            loss /= args.backprop_every

            loss.backward()
            optimizer.step()
            # train data eval
            with torch.no_grad():
                model = model.eval()
                #pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
                pred_score = np.concatenate([(pos_prob.sigmoid()).cpu().detach().numpy(), (neg_prob.sigmoid()).cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                # scaler = MinMaxScaler()
                # preds = np.transpose(scaler.fit_transform(np.transpose([pred_score])))[0]
                # pred_label = preds > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label.astype(np.float64), pred_score.astype(np.float64)))
                f1.append(f1_score(true_label.astype(np.float64), pred_label.astype(np.float64)))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label.astype(np.float64), pred_score.astype(np.float64)))
            # m_loss.append(loss.item())

            # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            if USE_MEMORY:
                model.memory.detach_memory()

        train_epoch_time = time.time() - train_start_epoch
        epoch_times.append(train_epoch_time)

        train_losses.append(np.mean(m_loss))
        # val_ap = 0.

        logger.info('Run {}, epoch: {}'.format(i, epoch))
        logger.info('train time: {}'.format(train_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train acc: {}'.format(np.mean(acc)))
        logger.info('train auc: {}'.format(np.mean(auc)))
        logger.info('train f1: {}'.format(np.mean(f1)))
        logger.info('train ap: {}'.format(np.mean(ap)))

        # recommendation 
        # if ((epoch + 1) % 20 == 0 and (epoch + 1) >= 200) or (epoch + 1) == args.n_epoch:
            # torch.save(
            #     {
            #         'model_state_dict': tgn.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': np.mean(m_loss),
            #         }, SAVE_MODEL_PATH
            # )
            # print("model saved")
        
        model.set_neighbor_finder(full_ngh_finder)
        val_ap, val_auc = eval_edge_prediction(model=model,
                                                negative_edge_sampler=val_rand_sampler,
                                                data=val_data,
                                                n_neighbors=NUM_NEIGHBORS)
        logger.info('valid ap: {}'.format(val_ap))
        logger.info('valid auc: {}'.format(np.mean(val_auc)))

        if USE_MEMORY:
            val_memory_backup = model.memory.backup_memory()

        if ((epoch + 1) % 5 == 0) or (epoch + 1) == args.n_epoch:  #  and epoch >= 30, epoch % 5 == 0 and epoch >= 30
            model.set_neighbor_finder(full_ngh_finder)
            # valid_start_time = time.time()
        
            # valid_result, valid_pred_output = eval_users(model, val_data.sources, val_data.destinations, val_data.timestamps, val_data.lats, val_data.lons, val_data.edge_idxs,
            #                                             train_data.sources, train_data.destinations, args)
            # valid_end_time = time.time()
            # valid_epoch_time = valid_end_time - valid_start_time
            # # print('valid: ', valid_result)
            # print('valid precision: ', valid_result['precision'], flush=True)
            # print('valid recall: ', valid_result['recall'], flush=True)
            # print('valid ndcg: ', valid_result['ndcg'], flush=True)
            # print('valid hit_ratio: ', valid_result['hit_ratio'], flush=True)
            # print('valid auc: ', valid_result['auc'], flush=True)
            # print('valid mrr: ', valid_result['mrr'], flush=True)
            # print('valid time: ', valid_epoch_time, flush=True)

            # if USE_MEMORY:
            #     val_memory_backup = model.memory.backup_memory()
            
            test_start_time = time.time()
            test_result, test_pred_output = eval_users(model, test_data.sources, test_data.destinations, test_data.timestamps, test_data.lats, test_data.lons, test_data.edge_idxs,
                                                    train_data.sources, train_data.destinations, args)
            test_end_time = time.time()
            test_epoch_time = test_end_time - test_start_time
            # print('test: ', test_result)
            print('test precision: ', test_result['precision'], flush=True)
            print('test recall: ', test_result['recall'], flush=True)
            print('test ndcg: ', test_result['ndcg'], flush=True)
            print('test hit_ratio: ', test_result['hit_ratio'], flush=True)
            print('test auc: ', test_result['auc'], flush=True)
            print('test mrr: ', test_result['mrr'], flush=True)
            print('test time: ', test_epoch_time, flush=True)

            # if best_mrr < valid_result['mrr']:
            #     best_mrr = valid_result['mrr']
            #     # best_result = test_result.copy()
            #     best_val_result = valid_result.copy()
            #     # best_outputs = test_pred_output

            if best_mrr < test_result['mrr']:
                best_mrr = test_result['mrr']
                best_result = test_result.copy()
                # best_val_result = valid_result.copy()
                best_outputs = test_pred_output
            
            # pickle.dump(best_outputs, open(f'{args.model_path}/test_outputs_{DATA}.pkl', "wb"))

        if USE_MEMORY:
            model.memory.restore_memory(val_memory_backup)  # memory restore
        # torch.save(model.state_dict(), get_checkpoint_path(epoch))

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
        val_memory_backup = model.memory.backup_memory()


    print('best test precision: ', best_result['precision'], flush=True)
    print('best test recall: ', best_result['recall'], flush=True)
    print('best test ndcg: ', best_result['ndcg'], flush=True)
    print('best test hit_ratio: ', best_result['hit_ratio'], flush=True)
    print('best test auc: ', best_result['auc'], flush=True)
    print('best test mrr: ', best_result['mrr'], flush=True)

    best_result_precision.append(best_result['precision'])
    best_result_recall.append(best_result['recall'])
    best_result_ndcg.append(best_result['ndcg'])
    best_result_hit.append(best_result['hit_ratio'])
    best_result_mrr.append(best_result['mrr'])

    # print('best valid precision: ', best_val_result['precision'], flush=True)
    # print('best valid recall: ', best_val_result['recall'], flush=True)
    # print('best valid ndcg: ', best_val_result['ndcg'], flush=True)
    # print('best valid hit_ratio: ', best_val_result['hit_ratio'], flush=True)
    # print('best valid auc: ', best_val_result['auc'], flush=True)
    # print('best valid mrr: ', best_val_result['mrr'], flush=True)

    # best_val_result_precision.append(best_val_result['precision'])
    # best_val_result_recall.append(best_val_result['recall'])
    # best_val_result_ndcg.append(best_val_result['ndcg'])
    # best_val_result_hit.append(best_val_result['hit_ratio'])
    # best_val_result_mrr.append(best_val_result['mrr'])

    logger.info('Saving LSTDR model')
    if USE_MEMORY:
        # Restore memory at the end of validation (save a model which is ready for testing)
        model.memory.restore_memory(val_memory_backup)
    # torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info('LSTDR model saved')

best_result_precision_mean, best_result_precision_std = np.mean(np.array(best_result_precision), axis=0), np.std(np.array(best_result_precision), axis=0)
best_result_recall_mean, best_result_recall_std = np.mean(np.array(best_result_recall), axis=0), np.std(np.array(best_result_recall), axis=0)
best_result_ndcg_mean, best_result_ndcg_std = np.mean(np.array(best_result_ndcg), axis=0), np.std(np.array(best_result_ndcg), axis=0)
best_result_hit_mean, best_result_hit_std = np.mean(np.array(best_result_hit), axis=0), np.std(np.array(best_result_hit), axis=0)
best_result_mrr_mean, best_result_mrr_std = np.mean(np.array(best_result_mrr)), np.std(np.array(best_result_mrr))

# best_val_result_precision_mean, best_val_result_precision_std = np.mean(np.array(best_val_result_precision), axis=0), np.std(np.array(best_val_result_precision), axis=0)
# best_val_result_recall_mean, best_val_result_recall_std = np.mean(np.array(best_val_result_recall), axis=0), np.std(np.array(best_val_result_recall), axis=0)
# best_val_result_ndcg_mean, best_val_result_ndcg_std = np.mean(np.array(best_val_result_ndcg), axis=0), np.std(np.array(best_val_result_ndcg), axis=0)
# best_val_result_hit_mean, best_val_result_hit_std = np.mean(np.array(best_val_result_hit), axis=0), np.std(np.array(best_val_result_hit), axis=0)
# best_val_result_mrr_mean, best_val_result_mrr_std = np.mean(np.array(best_val_result_mrr)), np.std(np.array(best_val_result_mrr))

print(f'Final test precision: {best_result_precision_mean} ± {best_result_precision_std}', flush=True)
print(f'Final test recall: {best_result_recall_mean} ± {best_result_recall_std}', flush=True)
print(f'Final test ndcg: {best_result_ndcg_mean} ± {best_result_ndcg_std}', flush=True)
print(f'Final test hit_ratio: {best_result_hit_mean} ± {best_result_hit_std}', flush=True)
print(f'Final test mrr: {best_result_mrr_mean} ± {best_result_mrr_std}', flush=True)

# print(f'Final valid precision: {best_val_result_precision_mean} ± {best_val_result_precision_std}', flush=True)
# print(f'Final valid recall: {best_val_result_recall_mean} ± {best_val_result_recall_std}', flush=True)
# print(f'Final valid ndcg: {best_val_result_ndcg_mean} ± {best_val_result_ndcg_std}', flush=True)
# print(f'Final valid hit_ratio: {best_val_result_hit_mean} ± {best_val_result_hit_std}', flush=True)
# print(f'Final valid mrr: {best_val_result_mrr_mean} ± {best_val_result_mrr_std}', flush=True)