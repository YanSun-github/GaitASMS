# # **************** For CASIA-B ****************
# # myModel
#CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 GaitASMS/main.py --cfgs ./configs/mymodel/mymodel.yaml --phase train

# # **************** For OUMVLP ****************
# # myModel
 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 GaitASMS/main.py --cfgs ./configs/mymodel/mymodel_OUMVLP.yaml --phase train