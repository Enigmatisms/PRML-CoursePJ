# res architecture baseline lr should be cut by half (along with batchsize)
CUDA_VISIBLE_DEVICES=0 python3 ./motif_mining.py --atcg_len 1000 --eval --load_model --load_path chkpt_2_res_bb_1000.pt