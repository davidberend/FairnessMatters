CUDA_VISIBLE_DEVICES=1 \
nohup /home/david/anaconda3/envs/david37/bin/python train.py -datafolder ./data/original -opt adam -train_path balanced_train.tsv -test_path balanced_test.tsv -model_name alexnet -num_epoches 50 -lr 0.0001 -pretrained_model /mnt/nvme/aibias/model_weights/pretrained/ImageNet_alexnet_ImageNet_adam_0.0001  2>&1 >./train_logs/balanced_alexnet_adam_0.0001_50.log & 
