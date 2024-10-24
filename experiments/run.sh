python -u run.py --config configs/imnet-r_all.yaml --gpuid 1 --repeat 5 --overwrite 1 --learner_type My_learner --learner_name All_feature  --log_dir outputs/imagenet-R/10-task/All_features_first_tuning12all_merge --feature_root imgnetR_feature_buffer.pt

python -u run.py --config configs/cifar-100_all.yaml --gpuid 1 --repeat 5 --overwrite 1 --learner_type My_learner --learner_name All_feature  --log_dir outputs/cifar-100/10-task/All_features_first_tuning12all_merge --feature_root cifar100_feature_buffer.pt

python -u run.py --config configs/imnet-100_all.yaml --gpuid 1 --repeat 5 --overwrite 1 --learner_type My_learner --learner_name All_feature  --log_dir outputs/imagenet-100/10-task/All_features_first_tuning12all_merge --feature_root imgnet100_feature_buffer.pt

python -u run.py --config configs/skin40_all.yaml --gpuid 1 --repeat 5 --overwrite 1 --learner_type My_learner --learner_name All_feature  --log_dir outputs/skin-40/4-task/All_features_first_tuning12all --feature_root skin40_feature_buffer.pt

python -u run.py --config configs/skin40_all_8task.yaml --gpuid 1 --repeat 5 --overwrite 1 --learner_type My_learner --learner_name All_feature  --log_dir outputs/skin-40/8-task/All_features_first_tuning12all --feature_root skin40_feature_buffer.pt

python -u run.py --config configs/skin40_all_20task.yaml --gpuid 1 --repeat 5 --overwrite 1 --learner_type My_learner --learner_name All_feature  --log_dir outputs/skin-40/20-task/All_features_first_tuning12all --feature_root skin40_feature_buffer.pt