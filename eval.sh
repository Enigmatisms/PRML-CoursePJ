CUDA_VISIBLE_DEVICES=1 python3 ./train_conv.py --atcg_len 1000 \
    --load_path pro_arch_new_1000.pt \
    --pos_threshold 0.5 --test_batch_size 100 --test_batches 100 \
    -e --load_model