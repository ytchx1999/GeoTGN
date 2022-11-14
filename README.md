# GeoTGN: Geographical Temporal Graph Network for Local-Life Service Recommendation

## 1. Overview
This repository is the official implementation of GeoTGN.

<!-- xxxx.

It contains an implementation of LSTDR in PyTorch, as well as multiple dynamic graph embedding approaches including:

- JODIE [[1]](https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf)
- DyREP [[2]](https://par.nsf.gov/servlets/purl/10099025)
- TGAT/TGSRec [[3]](https://arxiv.org/pdf/2002.07962.pdf), [[4]](https://arxiv.org/pdf/2108.06625.pdf)

The implementation of other baselines (NGCF [[5]](https://arxiv.org/pdf/1905.08108.pdf), LightGCN [[6]](https://arxiv.org/pdf/2002.02126.pdf), SASRec [[7]](https://arxiv.org/pdf/1808.09781.pdf?ref=https://githubhelp.com), TiSASRec [[8]](https://dl.acm.org/doi/pdf/10.1145/3336191.3371786?ref=https://githubhelp.com)) in our paper is available at：[https://github.com/ytchx1999/LSTDR-baseline](https://github.com/ytchx1999/LSTDR-baseline). -->

## 2. Setup 

### 2.1 Environment

Dependencies: 
```{bash}
python >= 3.7
torch == 1.8.2+cu102
pandas == 1.4.1
sklearn == 1.0.2
tqdm
```

GPU: NVIDIA A100 (80GB)

### 2.2 Dataset

There are two datasets can be used:
- [Gowalla-Food](http://www.yongliu.org/datasets.html) (Public)
- Meituan (Industrial)

## 3. Usage

### 3.1 Data Preprocessing

To run this code on the datasets, please first run the script to preprocess the data.

```bash
# gowalla
python3 utils/preprocess_gowalla.py --data gowalla --bipartite
# meituan_big
python3 utils/preprocess_meituan_big.py --data meituan_big --bipartite
```


### 3.2 Model Training and Inference

The `scripts/` folder contains training and inference scripts for models:

```bash
cd scripts/
# ${dataset} = [gowalla, meituan_big]
bash train_${dataset}.sh
```


<!-- ### 3.4 General Flags


```{bash}
  -d DATA, --data DATA  Dataset name (eg. wikipedia or reddit)
  --bs BS               Batch_size
  --prefix PREFIX       Prefix to name the checkpoints
  --n_degree N_DEGREE   Number of neighbors to sample
  --n_head N_HEAD       Number of heads used in attention layer
  --n_epoch N_EPOCH     Number of epochs
  --n_layer N_LAYER     Number of network layers
  --lr LR               Learning rate
  --patience PATIENCE   Patience for early stopping
  --n_runs N_RUNS       Number of runs
  --drop_out DROP_OUT   Dropout probability
  --gpu GPU             Idx for the gpu to use
  --node_dim NODE_DIM   Dimensions of the node embedding
  --time_dim TIME_DIM   Dimensions of the time embedding
  --backprop_every BACKPROP_EVERY
                        Every how many batches to backprop
  --use_memory          Whether to augment the model with a node memory
  --embedding_module {graph_attention,graph_sum,identity,time}
                        Type of embedding module
  --message_function {mlp,identity}
                        Type of message function
  --memory_updater {gru,rnn}
                        Type of memory updater
  --aggregator AGGREGATOR
  --memory_update_at_end
                        Whether to update memory at the end or at the start of the batch
  --message_dim MESSAGE_DIM
                        Dimensions of the messages
  --memory_dim MEMORY_DIM
                        Dimensions of the memory for each user
  --different_new_nodes
                        Whether to use disjoint set of new nodes for train and val
  --uniform             take uniform sampling from temporal neighbors
  --randomize_features  Whether to randomize node features
  --use_destination_embedding_in_message
                        Whether to use the embedding of the destination node as part of the message
  --use_source_embedding_in_message
                        Whether to use the embedding of the source node as part of the message
  --dyrep               Whether to run the dyrep model
  --pos_encode          Whether to run the pos encoding
  --reg REG             regularization
  --negsampleeval NEGSAMPLEEVAL
                        number of negative sampling evaluation, -1 for all
  --start_percent START_PERCENT
                        regularization
  --seed SEED           Seed for all
  --loss LOSS           Loss function
  --no_norm             Whether to use LayerNorm in MergeLayer
``` -->

<!-- ## Citation

If you find this code useful, please cite the following paper:
 
```bibtex
@inproceedings{tgn_icml_grl2020,
    title={Temporal Graph Networks for Deep Learning on Dynamic Graphs},
    author={Emanuele Rossi and Ben Chamberlain and Fabrizio Frasca and Davide Eynard and Federico 
    Monti and Michael Bronstein},
    booktitle={ICML 2020 Workshop on Graph Representation Learning},
    year={2020}
}
``` -->

## Note
The implemention is based on [Temporal Graph Networks](https://github.com/twitter-research/tgn).

<!-- ## References

[1] [Kumar S, Zhang X, Leskovec J. Predicting dynamic embedding trajectory in temporal interaction networks[C]//Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining. 2019: 1269-1278.](https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf)

[2] [Trivedi R, Farajtabar M, Biswal P, et al. Dyrep: Learning representations over dynamic graphs[C]//International conference on learning representations. 2019.](https://par.nsf.gov/servlets/purl/10099025)

[3] [Xu D, Ruan C, Korpeoglu E, et al. Inductive representation learning on temporal graphs[J]. arXiv preprint arXiv:2002.07962, 2020.](https://arxiv.org/pdf/2002.07962.pdf)

[4] [Fan Z, Liu Z, Zhang J, et al. Continuous-time sequential recommendation with temporal graph collaborative transformer[C]//Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021: 433-442.](https://arxiv.org/pdf/2108.06625.pdf)

[5] [Wang X, He X, Wang M, et al. Neural graph collaborative filtering[C]//Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval. 2019: 165-174.](https://arxiv.org/pdf/1905.08108.pdf)

[6] [He X, Deng K, Wang X, et al. Lightgcn: Simplifying and powering graph convolution network for recommendation[C]//Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 2020: 639-648.](https://arxiv.org/pdf/2002.02126.pdf)

[7] [Kang W C, McAuley J. Self-attentive sequential recommendation[C]//2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018: 197-206.](https://arxiv.org/pdf/1808.09781.pdf?ref=https://githubhelp.com)

[8] [Li J, Wang Y, McAuley J. Time interval aware self-attention for sequential recommendation[C]//Proceedings of the 13th international conference on web search and data mining. 2020: 322-330.](https://dl.acm.org/doi/pdf/10.1145/3336191.3371786?ref=https://githubhelp.com) -->

