"""
    Get Argument Parser
    @author: Qianyue He
    @date: 2022-11-25
"""

import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    # training parameters
    parser.add_argument("--seed", type = int, default = 0, help = "Random seed to use")
    parser.add_argument("--full_epochs", type = int, default = 500, help = "Training lasts for . epochs (wo warmup and cooldown)")
    parser.add_argument("--warmup_epochs", type = int, default = 10, help = "Warm up initialization epoch number")
    parser.add_argument("--cooldown_epochs", type = int, default = 100, help = "Epochs for lr cooldown")
    parser.add_argument("--train_eval_time", type = int, default = 1, help = "Evaluate every <x> epoch(s) during training")
    parser.add_argument("--max_ckpt_num", type = int, default = 3, help = "Maximum number of checkpoint that can be stored")
    parser.add_argument("--train_verbose", type = int, default = 0, help = "Whether to output anything within an epoch (0 means no, x > 0 means output every x batches)")

    # dataset & loader parameters
    parser.add_argument("--load_path", type = str, default = "", help = "Load path of model (or checkpoint), required when eval")
    parser.add_argument("--atcg_len", type = int, default = 1000, help = "Sequence number of a dataset")
    parser.add_argument("--batch_size", type = int, default = 100, help = "Train set batch size")
    parser.add_argument("--num_workers", type = int, default = 8, help = "Number of workers when loading the data")

    # evaluation parameters
    parser.add_argument("--test_batch_size", type = int, default = 100, help = "Test set batch size")
    parser.add_argument("--test_batches", type = int, default = 20, help = "Only to evaluate the first <test_batches> test batches (to save time)")

    # model parameters
    parser.add_argument("--emb_dropout", type = float, default = 0.1, help = "Convolutional embedding dropout")
    parser.add_argument("--mlp_dropout", type = float, default = 0.1, help = "Dropout for transformer MLP")
    parser.add_argument("--path_dropout", type = float, default = 0.1, help = "Random path dropout rate")
    parser.add_argument("--adam_wdecay", type = float, default = 0.1, help = "Weight decay for AdamW")

    # asymmetrical loss parameters
    # TODO: defaults should be modified
    parser.add_argument("--asl_gamma_pos", type = float, default = 0.02, help = "asl gamma pos")
    parser.add_argument("--asl_gamma_neg", type = float, default = 0.02, help = "asl gamma neg")
    parser.add_argument("--asl_eps", type = float, default = 0.02, help = "asl epsilon")
    parser.add_argument("--asl_clip", type = float, default = 0.02, help = "asl clipping")

    # lr scheduler parameters
    parser.add_argument("--cosine_folds", type = int, default = 4, help = "How many periods the cosine scheduler should have")
    parser.add_argument("--lr_max_start", type = float, default = 0.04, help = "The starting lr of upper bound lr")
    parser.add_argument("--lr_max_end", type = float, default = 0.006, help = "The ending lr of upper bound lr")
    parser.add_argument("--lr_min_start", type = float, default = 0.006, help = "The starting lr of lower bound lr")
    parser.add_argument("--lr_min_end", type = float, default = 0.002, help = "The ending lr of lower bound lr")

    # other parameters
    parser.add_argument("--pos_threshold", type = float, default = 0.5, help = "proba bigger than <threshold> will be classified as positive")

    # bool flags
    parser.add_argument("-b", "--debug", default = False, action = "store_true", help = "Code debugging (detect gradient anomaly and NaNs)")
    parser.add_argument("-d", "--del_dir", default = False, action = "store_true", help = "Delete dir ./logs and start new tensorboard records")
    parser.add_argument("-e", "--eval", default = False, action = "store_true", help = "Evaluation only model (a trained model is required)")
    parser.add_argument("-o", "--half_opt", default = False, action = "store_true", help = "Use AMP scaler to speed up")
    parser.add_argument("--load_model", default = False, action = "store_true", help = "Load from model path or load from checkpoint path")
    return parser.parse_args()