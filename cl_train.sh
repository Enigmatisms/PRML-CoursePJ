if [ ! -d "./check_points/" ]; then
    echo "Folder './check_points/' not found, creating folder './check_points/'"
    mkdir ./check_points/
fi

if [ ! -d "./model/" ]; then
    echo "Folder './model/' not found, creating folder './model/'"
    mkdir ./model/
fi

CUDA_VISIBLE_DEVICES=1 python3 ./train_moco.py --atcg_len 1000 --adam_wdecay 1e-3 \
    --lr_max_start 2e-3 --lr_max_end 2e-4 --lr_min_start 1e-4 --lr_min_end 5e-5 \
    --train_eval_time 300 \
    --train_verbose 0 --exp_name moco -n