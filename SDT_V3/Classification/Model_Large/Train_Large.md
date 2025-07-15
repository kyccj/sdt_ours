### Results on Imagenet-1K

Trained weights of 171M_1x4: [here](https://drive.google.com/file/d/1sJAjirbjVaB7gLSybvy2Xz2wwQl6gZk7/view?usp=sharing).

Trained weights of 171M_1x8: [here](https://drive.google.com/file/d/18bcS2jQD41JyoJAW9lhZOkTgUb79uShf/view?usp=sharing).

Trained weights of  171M_1x8_384: [here](https://drive.google.com/file/d/1ooNGJRTi869e0ApZm8Oc84Mq02uXXyA8/view?usp=sharing).


Trained weights of 83M_1x4: [here](https://drive.google.com/file/d/1f9pFflYcMacnYJc2u8cHcgMqdibv8wAO/view?usp=sharing).

Trained weights of 83M_1x8: [here](https://drive.google.com/file/d/1sh4F9LWFbKIgIVa2u0QaixBWIcbDZ7h_/view?usp=sharing).

Trained weights of  83M_1x8_384: [here]().

### Data Prepare

ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Pretrain & Finetune

Pretrain:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  main_pretrain.py \
  --batch_size 256 \
  --blr 1.5e-4 \
  --warmup_epochs 20 \
  --epochs 200 \
  --model spikmae_12_512 \
  --mask_ratio 0.50 \
  --data_path ../imagenet1-k \
```

Finetune:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,6,7,8 torchrun --standalone --nproc_per_node=8 \
 main_finetune.py \
  --batch_size 100 \
  --blr 6e-4 \
  --warmup_epochs 10 \
  --layer_decay 0.75 \
  --finetune ../171M-1x8_86_2.pth\
  --epochs 150 \
  --drop_path 0.1 \
  --model spikformer12_768 \
  --data_path /mnt/hdd1/kyccj/data \
  --output_dir /mnt/hdd1/kyccj/H-direct_new_base/ImageNet_SDTv3_large/ours/171M_1x8 \
  --log_dir /mnt/hdd1/kyccj/H-direct_new_base/ImageNet_SDTv3_large/ours/171M_1x8 \
  --reprob 0.25 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --dist_eval
```

Distillation:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 196 \
  --blr 1e-3 \
  --warmup_epochs 5 \
  --epochs 100 \
  --drop_path 0.1 \
  --finetune finetune_checkpoint.pth \
  --model spikformer12_512 \
  --data_path ../imagenet1-k \
  --output_dir ./outputs/.. \
  --log_dir ./outputs/.. \
  --dist_eval \
  --time_steps 1 \
  --kd \
  --input_size 224 \
  --teacher_model caformer_b36_in21ft1k \
  --reprob 0.25 \
  --mixup 0.5 \
  --cutmix 1.0 \
  --distillation_type hard 
```



