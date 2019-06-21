"""
Groups the kwargs in the config file into relevant groups that can be
cleanly passed to other functions.
"""


def imagedata_kwargs(parsed_args):
    return {
        'root': parsed_args["root"],
        'sources': parsed_args["sources"],
        'targets': parsed_args["targets"],
        'height': parsed_args["height"],
        'width': parsed_args["width"],
        'random_erase': parsed_args["random_erase"],
        'color_jitter': parsed_args["color_jitter"],
        'color_aug': parsed_args["color_aug"],
        'use_cpu': parsed_args["use_cpu"],
        'split_id': parsed_args["split_id"],
        'combineall': parsed_args["combineall"],
        'batch_size': parsed_args["batch_size"],
        'workers': parsed_args["workers"],
        'num_instances': parsed_args["num_instances"],
        'train_sampler': parsed_args["train_sampler"],
        # image
        'cuhk03_labeled': parsed_args["cuhk03_labeled"],
        'cuhk03_classic_split': parsed_args["cuhk03_classic_split"],
        'market1501_500k': parsed_args["market1501_500k"],
        # new
        'val_split': parsed_args["val_split"],
        'seed': parsed_args["seed"]
    }


def videodata_kwargs(parsed_args):
    return {
        'root': parsed_args["root"],
        'sources': parsed_args["sources"],
        'targets': parsed_args["targets"],
        'height': parsed_args["height"],
        'width': parsed_args["width"],
        'random_erase': parsed_args["random_erase"],
        'color_jitter': parsed_args["color_jitter"],
        'color_aug': parsed_args["color_aug"],
        'use_cpu': parsed_args["use_cpu"],
        'split_id': parsed_args["split_id"],
        'combineall': parsed_args["combineall"],
        'batch_size': parsed_args["batch_size"],
        'workers': parsed_args["workers"],
        'num_instances': parsed_args["num_instances"],
        'train_sampler': parsed_args["train_sampler"],
        # video
        'seq_len': parsed_args["seq_len"],
        'sample_method': parsed_args["sample_method"],
        # new
        'val_split': parsed_args["val_split"],
        'seed': parsed_args["seed"]
    }


def optimizer_kwargs(parsed_args):
    return {
        'optim': parsed_args["optim"],
        'lr': parsed_args["lr"],
        'weight_decay': parsed_args["weight_decay"],
        'momentum': parsed_args["momentum"],
        'sgd_dampening': parsed_args["sgd_dampening"],
        'sgd_nesterov': parsed_args["sgd_nesterov"],
        'rmsprop_alpha': parsed_args["rmsprop_alpha"],
        'adam_beta1': parsed_args["adam_beta1"],
        'adam_beta2': parsed_args["adam_beta2"],
        'staged_lr': parsed_args["staged_lr"],
        'new_layers': parsed_args["new_layers"],
        'base_lr_mult': parsed_args["base_lr_mult"]
    }


def lr_scheduler_kwargs(parsed_args):
    return {
        'lr_scheduler': parsed_args["lr_scheduler"],
        'stepsize': parsed_args["stepsize"],
        'gamma': parsed_args["gamma"],
        'lr_sched_patience': parsed_args["lr_sched_patience"],
        'lr_sched_threshold': parsed_args["lr_sched_threshold"],
        'lr_sched_cooldown': parsed_args["lr_sched_cooldown"]
    }


def engine_run_kwargs(parsed_args):
    return {
        'save_dir': parsed_args["save_dir"],
        'max_epoch': parsed_args["max_epoch"],
        'start_epoch': parsed_args["start_epoch"],
        'fixbase_epoch': parsed_args["fixbase_epoch"],
        'open_layers': parsed_args["open_layers"],
        'start_eval': parsed_args["start_eval"],
        'eval_freq': parsed_args["eval_freq"],
        'test_only': parsed_args["evaluate"],
        'print_freq': parsed_args["print_freq"],
        'dist_metric': parsed_args["dist_metric"],
        'normalize_feature': parsed_args["normalize_feature"],
        'visrank': parsed_args["visrank"],
        'visrank_topk': parsed_args["visrank_topk"],
        'use_metric_cuhk03': parsed_args["use_metric_cuhk03"],
        'ranks': parsed_args["ranks"],
        'rerank': parsed_args["rerank"]
    }