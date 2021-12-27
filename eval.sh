python -m torch.distributed.launch --nproc_per_node 1 --master_port 12346 main.py --eval \
--amp-opt-level O0 --cfg configs/nommer_small_attn.yaml --batch-size 128 \
--resume nommer_small.pth --data-path your-imagenet-path