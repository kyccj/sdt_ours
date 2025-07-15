### Results on Imagenet-1K

Trained weights of 5.1M: [here](https://drive.google.com/file/d/1LMkOTPehDNpQE79bvB7jFTf6UzDjpAHQ/view?usp=drive_link).

Trained weights of 10M: [here](https://drive.google.com/file/d/1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq/view?usp=sharing).

Trained weights of  19M: [here](https://drive.google.com/file/d/1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H/view?usp=drive_link).

Others weights are coming soon.
### Train 

Train:

```shell
CUDA_VISIBLE_DEVICES=4,5,6,9 torchrun --standalone --nproc_per_node=4 \
  main_finetune.py \
  --batch_size 512 \
  --blr 6e-4 \
  --warmup_epochs 5 \
  --epochs 200 \
  --model Efficient_Spiking_Transformer_m \
  --data_path /mnt/hdd1/kyccj/ImageNet_down \
  --output_dir /mnt/hdd1/kyccj/H-direct_new_base/ImageNet_SDTv3/ours/1.0 \
  --log_dir /mnt/hdd1/kyccj/H-direct_new_base/ImageNet_SDTv3/ours/1.0 \
  --model_mode ms \
  --dist_eval
```

  --master_port=29601\
  --rdzv_endpoint=localhost:29501\
  --rdzv_id=exp3\

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
