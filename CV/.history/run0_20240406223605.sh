#!/bin/bash
#!/bin/bash
python IP-remove_food.py --dataset cifar100 --noise_type noisy100 --repeat 3 --device cuda:1 --ensemble_size 5 --dropout_IP 0.01 --acc 56.41 >> cifar100_ensemble5.txt 2>&1
python IP-remove_food.py --dataset cifar100 --noise_type noisy100 --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 0.01 --acc 56.41 >> cifar100.txt 2>&1
python IP-remove_food.py --dataset cifar100 --noise_type noisy100 --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 0.01 --acc 56.41 --last_layer -- True >> cifar100_lastlayer.txt 2>&1


python IP-remove_food.py --dataset cifar100 --noise_type noisy100 --repeat 3 --device cuda:1 --ensemble_size 15 --dropout_IP 0.01 --acc 56.41 >> cifar100_ensemble15.txt 2>&1
python IP-remove_food.py --dataset cifar100 --noise_type noisy100 --repeat 3 --device cuda:1 --ensemble_size 10 --dropout_IP 0.01 --acc 56.41 >> cifar100_ensemble10.txt 2>&1

python IP-remove_food.py --dataset cifar100 --noise_type noisy100 --repeat 3 --device cuda:1 --ensemble_size 1 --dropout_IP 0.01 --acc 56.41 >> cifar100_ensemble1.txt 2>&1

python IP-remove_food.py --dataset cifar100 --noise_type noisy100 --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 0.1 --acc 56.41 >> cifar100_drop10%.txt 2>&1
python IP-remove_food.py --dataset cifar100 --noise_type noisy100 --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 0.5 --acc 56.41 >> cifar100_drop50%.txt 2>&1
python IP-remove_food.py --dataset cifar100 --noise_type noisy100 --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 1 --acc 56.41 >> cifar100_drop100s%.txt 2>&1