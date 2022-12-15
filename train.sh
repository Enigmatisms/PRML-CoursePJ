if [ ! -d "./check_points/" ]; then
    echo "Folder './check_points/' not found, creating folder './check_points/'"
    mkdir ./check_points/
fi

if [ ! -d "./model/" ]; then
    echo "Folder './model/' not found, creating folder './model/'"
    mkdir ./model/
fi

# CUDA_VISIBLE_DEVICES=1 python3 ./train_conv.py --atcg_len 1000 --adam_wdecay 2e-3 \
#     --lr_max_start 6e-4 --lr_max_end 5e-5 --lr_min_start 5e-4 --lr_min_end 2e-5 \
#     --pos_threshold 0.5 --emb_dropout 0.2 --class_dropout 0.2 --conv_dropout 0.2 --input_dropout 0.2 \
#     --batch_size 100 --test_batch_size 100 --test_batches 40 --train_eval_time 5 \
#     --train_verbose 100 --exp_name new_arch

# res architecture baseline lr should be cut by half (along with batchsize)
CUDA_VISIBLE_DEVICES=1 python3 ./train_conv.py --atcg_len -1 --adam_wdecay 2e-2 \
    --lr_max_start 2e-3 --lr_max_end 2e-4 --lr_min_start 1.5e-3 --lr_min_end 4e-5 \
    --pos_threshold 0.5 --emb_dropout 0.2 --class_dropout 0.25 --conv_dropout 0.3 --input_dropout 0.2 --path_dropout 0.0 \
    --batch_size 100 --test_batch_size 100 --test_batches 40 --train_eval_time 5 \
    --train_verbose 100 --exp_name res_unified