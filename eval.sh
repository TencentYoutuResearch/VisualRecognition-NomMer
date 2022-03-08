python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--amp-opt-level O1 --cfg configs/nommer_small_attn.yaml --batch-size 256 \
--resume <checkpoint> --data-path <imagenet-path>