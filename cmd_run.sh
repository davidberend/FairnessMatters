
CUDA_VISIBLE_DEVICES=1 \
nohup /home/david/anaconda3/envs/david37/bin/python train.py -datafolder ./data/original -opt adam -train_path train_new.tsv -test_path test_new.tsv -model_name resnet50 -num_epoches 100 -lr 0.0001 -pretrained_model /mnt/nvme/aibias/model_weights/pretrained/trained_resnet50_both_adam_0.001  2>&1 >./train_logs/croped_0.0001_100.log & 
