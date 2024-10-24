#bash experiments/cifar-100.sh
#bash experiments/imagenet-r.sh
#bash experiments/imagenet-r_short.sh
#bash experiments/imagenet-r_long.sh
#bash experiments/domainnet.sh
python -u run.py --config configs/PathMNIST_all.yaml --gpuid 0 --repeat 3 --overwrite 0 --learner_type My_learner --learner_name All_feature  --log_dir outputs/pathminst/4-task/All_features33_500 --feature_root pathmnist_feature_buffer.pt
python -u run.py --config configs/OrganSMNIST_all.yaml --gpuid 0 --repeat 3 --overwrite 0 --learner_type My_learner --learner_name All_feature  --log_dir outputs/organsminst/5-task/All_features33_500 --feature_root organsmnist_feature_buffer.pt

#python -u run.py --config configs/PathMNIST_all2.yaml --gpuid 1 --repeat 3 --overwrite 0 --learner_type My_learner --learner_name All_feature  --log_dir outputs/pathminst/4-task/All_features00 --feature_root pathmnist_feature_buffer.pt
#python -u run.py --config configs/OrganSMNIST_all2.yaml --gpuid 1 --repeat 3 --overwrite 0 --learner_type My_learner --learner_name All_feature  --log_dir outputs/organsminst/5-task/All_features00 --feature_root organsmnist_feature_buffer.pt


#python -u run.py --config configs/PathMNIST_all.yaml --gpuid 0 --repeat 1 --overwrite 1 --learner_type My_learner --learner_name All_feature  --log_dir outputs/pathminst/1-task/upper --feature_root pathmnist_feature_buffer.pt
#python -u run.py --config configs/OrganSMNIST_all.yaml --gpuid 0 --repeat 1 --overwrite 1 --learner_type My_learner --learner_name All_feature  --log_dir outputs/organsminst/1-task/upper --feature_root organsmnist_feature_buffer.pt
#python -u run.py --config configs/skin40_all.yaml --gpuid 0 --repeat 1 --overwrite 1 --learner_type My_learner --learner_name All_feature  --log_dir outputs/skin40/1-task/upper --feature_root skin40_feature_buffer.pt
