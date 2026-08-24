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



import spikformer
import spikformer_s_direct
import spikformer_a2sg
import spike_reg

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
    if spike_reg.REG.enabled:
        # lambda is O(1e-8); the default {:.4f} format would print it as 0.0000
        metric_logger.add_meter("lam", misc.SmoothedValue(window_size=1, fmt="{value:.3e}"))
        metric_logger.add_meter("reg_R", misc.SmoothedValue(fmt="{global_avg:.4g}"))
        metric_logger.add_meter("spikes", misc.SmoothedValue(fmt="{global_avg:.4g}"))
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

        # the wrapped neurons append their reg terms to REG during the forward pass,
        # so it has to be cleared before every one of them
        spike_reg.REG.reset()

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if args.kd:
                loss = criterion(samples, outputs, targets)
                outputs_acc, _ = outputs
            else:
                loss = criterion(outputs, targets)
                outputs_acc = outputs

            task_loss_value = loss.item()
            reg_R_value = 0.0
            spikes_value = 0.0
            if spike_reg.REG.enabled and spike_reg.REG.n_layers > 0:
                R = spike_reg.REG.total()
                # R is the RAW, pre-lambda value -- it is what the controller solves
                # lambda against, so it is logged before the multiplication
                reg_R_value = R.item()
                spikes_value = spike_reg.REG.spikes().item() / samples.shape[0]
                if spike_reg.REG.lam > 0.0:
                    loss = loss + spike_reg.REG.lam * R
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
        if spike_reg.REG.enabled:
            # global_avg of these meters is reduced across ranks by
            # synchronize_between_processes(), so the controller downstream sees the
            # same numbers on every rank and no broadcast of lambda is needed
            metric_logger.update(task_loss=task_loss_value)
            metric_logger.update(reg_R=reg_R_value)
            metric_logger.update(spikes=spikes_value)
            metric_logger.update(lam=spike_reg.REG.lam)
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
def evaluate(data_loader, model, device, model_mode="ms"):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    total_spike_count =0.0
    encod_spike_count = 0.0
    n_images_local = 0
    for batch in metric_logger.log_every(data_loader, 500, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            # Spike counting.  One generic pass instead of a branch per model_mode:
            # a wrapped neuron (EIPMultispike) reports through the wrapper, and its
            # inner counter -- which a2sg/s_direct keep -- is zeroed so it is not
            # counted twice.  Plain ms Multispike has no counter at all, which is why
            # the old `else` branch raised AttributeError; hasattr now guards it.
            wrapped_inner = set()
            for m in model.modules():
                if isinstance(m, spike_reg.EIPMultispike):
                    total_spike_count += m.spike_count_int.item()
                    m.spike_count_int.zero_()
                    wrapped_inner.add(id(m.inner))
                    if hasattr(m.inner, "spike_count_int"):
                        m.inner.spike_count_int.zero_()
            for m in model.modules():
                if isinstance(m, spike_reg.EIPMultispike) or id(m) in wrapped_inner:
                    continue
                if not isinstance(m, (spikformer.Multispike,
                                      spikformer_s_direct.Multispike,
                                      spikformer_s_direct.Multispike_first,
                                      spikformer_a2sg.Multispike)):
                    continue
                if hasattr(m, "spike_count_int"):
                    total_spike_count += m.spike_count_int.item()
                    m.spike_count_int.zero_()
                if hasattr(m, "spike_count_int_encod"):
                    encod_spike_count += m.spike_count_int_encod.item()
                    m.spike_count_int_encod.zero_()
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # functional.reset_net(model)

        batch_size = images.shape[0]
        n_images_local += batch_size
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
    # per image, over the images THIS rank saw (the counters are per-rank).  The
    # hard-coded 50000 was wrong for anything but full ImageNet val on one process.
    n_images_local = max(n_images_local, 1)
    total_spike_count = total_spike_count / n_images_local
    encod_spike_count = encod_spike_count / n_images_local
    print(f"\n Total spikes: {total_spike_count:.1f}")
    print(f"\n Encod spikes: {encod_spike_count:.1f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
