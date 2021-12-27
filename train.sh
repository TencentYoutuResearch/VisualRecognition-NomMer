python -m torch.distributed.launch \
--nproc_per_node 1 --master_port 12346  main.py \
--cfg ./configs/nommer_small_attn.yaml --data-path your-imagenet-path \
--amp-opt-level O1 --accumulation-steps 1 --batch-size 32 --output ./output
