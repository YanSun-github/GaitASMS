# # **************** For CASIA-B ****************
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 GaitASMS/main.py --cfgs ./configs/mymodel/mymodel.yaml --phase test


# # **************** For OUMVLP ****************
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 GaitASMS/main.py --cfgs ./configs/mymodel/mymodel_OUMVLP.yaml --phase test