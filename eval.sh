CUDA_VISIBLE_DEVICES=0 python3 ./train_conv.py --atcg_len -1 \
    --load_path chkpt_2_res_bb_1000.pt \
    --pos_threshold 0.5 --test_batch_size 50 --test_batches 200 \
    -e --load_model