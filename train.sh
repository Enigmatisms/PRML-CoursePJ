if [ ! -d "./check_points/" ]; then
    echo "Folder './check_points/' not found, creating folder './check_points/'"
    mkdir ./check_points/
fi

if [ ! -d "./model/" ]; then
    echo "Folder './model/' not found, creating folder './model/'"
    mkdir ./model/
fi

CUDA_VISIBLE_DEVICES=0 python3 ./train_swin.py --atcg_len 1000 --adam_wdecay 2e-3 \
    --lr_max_start 2e-3 --lr_max_end 2e-4 --lr_min_start 5e-4 --lr_min_end 5e-5 \
    --pos_threshold 0.5 --emb_dropout 0.0 --mlp_dropout 0.0 --path_dropout 0.0 --class_dropout 0.0\
    --batch_size 50 --test_batch_size 50 --test_batches 40 --train_eval_time 5 \
    --train_verbose 100
