# CUDA_VISIBLE_DEVICES=0 python train.py --granularity day --train_forecaster --save_models
# CUDA_VISIBLE_DEVICES=0 python train.py --granularity hour --train_forecaster --train_classifier --save_models

CUDA_VISIBLE_DEVICES=0 python train.py --granularity day --train_forecaster --save_models --temperature_penalty 4 
CUDA_VISIBLE_DEVICES=0 python train.py --granularity hour --train_forecaster --save_models --temperature_penalty 4 