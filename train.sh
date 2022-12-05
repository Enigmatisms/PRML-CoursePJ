if [ ! -d "./check_points/" ]; then
    echo "Folder './check_points/' not found, creating folder './check_points/'"
    mkdir ./check_points/
fi

if [ ! -d "./model/" ]; then
    echo "Folder './model/' not found, creating folder './model/'"
    mkdir ./model/
fi

CUDA_VISIBLE_DEVICES=0 python3 ./train_swin.py --atcg_len 1000 --adam_wdecay 1e-3 \
    --lr_max_start 8e-4 --lr_max_end 8e-5 --lr_min_start 3e-4 --lr_min_end 3e-5 \
    --pos_threshold 0.5 --emb_dropout 0.05 --path_dropout 0.1 --class_dropout 0.15 \
    --batch_size 100 --test_batch_size 100 --test_batches 30 --train_eval_time 5 \
    --train_verbose 100 --seed 3407 --mixup 0.4 --mixup_epochs 400
