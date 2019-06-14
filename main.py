import sys
import os
import os.path as osp
import warnings
import time
import yaml
import argparse

import torch
import torch.nn as nn

from configs.get_config_kwargs import (
    imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs
)
import torchreid
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info
)



def build_datamanager(args):
    if args["app"] == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(args))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(args))


def build_engine(args, datamanager, model, optimizer, scheduler):
    if args["app"] == 'image':
        if args["loss"] == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_cpu=args["use_cpu"],
                label_smooth=args["label_smooth"]
            )
        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=args["margin"],
                weight_t=args["weight_t"],
                weight_x=args["weight_x"],
                scheduler=scheduler,
                use_cpu=args["use_cpu"],
                label_smooth=args["label_smooth"]
            )

    else:
        if args["loss"] == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_cpu=args["use_cpu"],
                label_smooth=args["label_smooth"],
                pooling_method=args["pooling_method"]
            )
        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=args["margin"],
                weight_t=args["weight_t"],
                weight_x=args["weight_x"],
                scheduler=scheduler,
                use_cpu=args["use_cpu"],
                label_smooth=args["label_smooth"]
            )

    return engine


def main():

    # Load parameters from config file
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='path to configuration file')
    args = parser.parse_args()
    with open(args.config, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    # Set random seeds
    set_random_seed(config["seed"])

    # Set up GPU
    if not config["use_avai_gpus"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = config["gpu_devices"]
    use_gpu = torch.cuda.is_available() and not config["use_cpu"]

    # Set up log files
    log_name = 'test.log' if config["evaluate"] else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(config["save_dir"], log_name))

    # Prepare for training
    print('==========\nArgs:{}\n=========='.format(config))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    if use_gpu:
        torch.backends.cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    # Build datamanager and model
    datamanager = build_datamanager(config)

    print('Building model: {}'.format(config["arch"]))
    model = torchreid.models.build_model(
        name=config["arch"],
        num_classes=datamanager.num_train_pids,
        loss=config["loss"].lower(),
        pretrained=(not config["no_pretrained"]),
        use_gpu=use_gpu
    )

    # Compute model complexity
    num_params, flops = compute_model_complexity(model, (1, 3, config["height"], config["width"]))
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    # Load pretrained weights if necessary
    if config["load_weights"] and check_isfile(config["load_weights"]):
        load_pretrained_weights(model, config["load_weights"])

    # Set up multi-gpu
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # Model settings
    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(config))
    scheduler = torchreid.optim.build_lr_scheduler(optimizer, **lr_scheduler_kwargs(config))

    if config["resume"] and check_isfile(config["resume"]):
        config["start_epoch"] = resume_from_checkpoint(config["resume"], model, optimizer=optimizer)

    print('Building {}-engine for {}-reid'.format(config["loss"], config["app"]))
    engine = build_engine(config, datamanager, model, optimizer, scheduler)

    engine.run(**engine_run_kwargs(config))


if __name__ == '__main__':
    main()