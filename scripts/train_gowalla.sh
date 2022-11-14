cd "$(dirname $0)"

python3 ../main.py \
-d gowalla \
--n_degree 20 \
--bs 600 \
--n_epoch 10 \
--n_layer 1 \
--lr 0.0001 \
--prefix lstdr \
--n_runs 1 \
--drop_out 0.1 \
--gpu 0 \
--node_dim 100 \
--time_dim 100 \
--message_dim 100 \
--memory_dim 100 \
--reg 0.1 \
--negsampleeval -1 \
--use_memory \
--seed 0 \
--max_dist 20. \
--data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/DATA4HOPE/local-tgn/ \
--model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/DATA4HOPE/local-tgn/model \
--log_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/DATA4HOPE/local-tgn/log


# --uniform

