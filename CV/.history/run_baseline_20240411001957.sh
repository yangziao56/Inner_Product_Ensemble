#!/bin/bash
python IP-remove_baselines.py --dataset cifar100 --noise_type noisy100 --acc 56.41 >> baseline/100n_baseline.txt 2>&1
python IP-remove_baselines.py --noise_type aggre --acc 91.62 >> baseline/aggre.txt 2>&1
python IP-remove_baselines.py --noise_type rand1 --acc 90.25 >> baseline/rand1.txt 2>&1
python IP-remove_baselines.py --noise_type worst --acc 85.66 >> baseline/worst.txt 2>&1

