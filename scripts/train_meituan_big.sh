cd "$(dirname $0)"

python3 ../main.py \
-d meituan_big \
--n_degree 30 \
--bs 200 \
--n_epoch 20 \
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
--data_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/DATA4HOPE/local-tgn/ \
--model_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/DATA4HOPE/local-tgn/model \
--log_path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/DATA4HOPE/local-tgn/log


# --uniform

