CUDA_VISIBLE_DEVICES=0 python3 ./train_swin.py --atcg_len 1000 --adam_wdecay 3e-2 \
    --lr_max_start 2e-3 --lr_max_end 2e-4 --lr_min_start 5e-4 --lr_min_end 5e-5 \
    --pos_threshold 0.5 --emb_dropout 0.05 --mlp_dropout 0.05 --path_dropout 0.05 --class_dropout 0.05\
    --batch_size 20 --test_batch_size 40 --test_batches 50 \
    --train_verbose 100 -d
