python -m torch.distributed.launch \
--nproc_per_node 8 --master_port 12345 main.py \
--cfg configs/nommer_small_attn.yaml --data-path <imagenet-path> \
--amp-opt-level O1 --accumulation-steps 1 --batch-size 128 --output ./output
