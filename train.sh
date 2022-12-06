if [ ! -d "./check_points/" ]; then
    echo "Folder './check_points/' not found, creating folder './check_points/'"
    mkdir ./check_points/
fi

if [ ! -d "./model/" ]; then
    echo "Folder './model/' not found, creating folder './model/'"
    mkdir ./model/
fi

<<<<<<< HEAD
CUDA_VISIBLE_DEVICES=1 python3 ./train_swin.py --atcg_len 1000 --adam_wdecay 1e-4 \
<<<<<<< HEAD
    --lr_max_start 3e-4 --lr_max_end 3e-5 --lr_min_start 1e-4 --lr_min_end 1e-5 \
    --pos_threshold 0.5 --emb_dropout 0.0 --mlp_dropout 0.0 --path_dropout 0.0 \
    --class_dropout 0.0 --att_dropout 0.0 --proj_dropout 0.0 --patch_pool 2 \
=======
    --lr_max_start 5e-4 --lr_max_end 5e-5 --lr_min_start 1e-4 --lr_min_end 1e-5 \
    --pos_threshold 0.5 --emb_dropout 0.1 --class_dropout 0.1 \
>>>>>>> Baseline retrain
    --batch_size 50 --test_batch_size 50 --test_batches 40 --train_eval_time 5 \
=======
CUDA_VISIBLE_DEVICES=1 python3 ./train_swin.py --atcg_len 1000 --adam_wdecay 5e-4 \
    --lr_max_start 6e-4 --lr_max_end 5e-5 --lr_min_start 2e-4 --lr_min_end 2e-5 \
    --pos_threshold 0.5 --emb_dropout 0.1 --class_dropout 0.1 --conv_dropout 0.1 \
    --batch_size 50 --test_batch_size 50 --test_batches 50 --train_eval_time 5 \
>>>>>>> New pure conv testing (BN + Drop)
    --train_verbose 100
