gpus=0
seed=-1
data="SYSUMM01"
dataset_short="MM01"
train_dataset="MARKET1501_TRANSFORMED_TO_${dataset_short}"
output_dir="logs/${dataset_short}"
output="${output_dir}/stage1"

python tools/train_stage1.py --config-file "configs/stage1_config.yml" \
--gpus ${gpus} \
--seed ${seed} \
OUTPUT_DIR ${output} \
DATASETS.NAMES "('${train_dataset}',)" \
DATASETS.TESTS "('${data}',)" \
MODEL.LOSSES.KL.SCALE 0.0 \

checkpoint="${output}/model_final.pth"
python tools/train_stage1.py --config-file "configs/stage1_config.yml" \
--gpus ${gpus} \
--seed ${seed} \
MODEL.WEIGHTS ${checkpoint} \
OUTPUT_DIR ${output} \
DATASETS.NAMES "('${train_dataset}',)" \
DATASETS.TESTS "('${data}',)" \
MODEL.LOSSES.KL.SCALE 0.1

cluster_method="CROSS_DBSCAN_BGM"
checkpoint_path="${output}/model_final.pth"
output="${output_dir}/stage2"
cluster_num=-1
save_pre="${cluster_method}"
python tools/train_stage2.py \
--config-file "configs/stage2_config.yml" \
--gpus ${gpus} \
--seed ${seed} \
--eps 0.4 \
SAVE_PRE ${save_pre} OUTPUT_DIR ${output} \
MODEL.WEIGHTS ${checkpoint_path} \
DATASETS.NAMES "('${data}',)" \
DATASETS.TESTS "('${data}',)" \
UL.CLUSTER.NUM ${cluster_num} \
UL.CLUSTER.METHOD ${cluster_method} \
UL.CLUSTER.TIMES -1

cluster_num=400
cluster_method="CROSS_KMEANS_BGM"
checkpoint_path="${output}/${save_pre}model_final.pth"
python tools/train_stage2.py \
--config-file "configs/stage2_config.yml" \
--gpus ${gpus} \
--seed ${seed} \
SAVE_PRE "${save_pre}_${cluster_method}" OUTPUT_DIR ${output} \
MODEL.WEIGHTS ${checkpoint_path} \
DATASETS.NAMES "('${data}',)" \
DATASETS.TESTS "('${data}',)" \
UL.CLUSTER.NUM ${cluster_num} \
UL.CLUSTER.METHOD ${cluster_method} \
UL.CLUSTER.TIMES -1