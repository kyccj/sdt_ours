# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch



import spikformer_s_direct

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched



# from spikingjelly.clock_driven import functional


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    max_norm,
    mixup_fn,
    log_writer,
    args,
    model_ema,
):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_nomix = targets
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if args.kd:
                loss = criterion(samples, outputs, targets)
                outputs_acc, _ = outputs
            else:
                loss = criterion(outputs, targets)
                outputs_acc = outputs
        # outputs_acc, _ = outputs
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model)
        torch.cuda.synchronize()
        batch_size = samples.shape[0]
        acc1, acc5 = accuracy(outputs_acc, targets_nomix, topk=(1, 5))
        # functional.reset_net(model)
        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        
#         cal_acc(metric_logger,outputs,targets)
        
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print(
        "* Train_Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},model_ema

def cal_acc(metric_logger,output,target):
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    return metric_logger.acc1,metric_logger.acc5
    
@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    total_spike_count =0.0
    encod_spike_count = 0.0
    for batch in metric_logger.log_every(data_loader, 500, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            for m in model.modules():
                if isinstance(m,spikformer_s_direct.Multispike_first):
                    total_spike_count += m.spike_count_int.item()
                    encod_spike_count += m.spike_count_int_encod.item()
                    m.spike_count_int.zero_()
                    m.spike_count_int_encod.zero_()
                if isinstance(m,spikformer_s_direct.Multispike):
                    total_spike_count += m.spike_count_int.item()
                    m.spike_count_int.zero_()
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # functional.reset_net(model)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    total_spike_count = total_spike_count / 50000
    encod_spike_count = encod_spike_count / 50000
    print(f"\n Total spikes: {total_spike_count:.1f}")
    print(f"\n Encod spikes: {encod_spike_count:.1f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
