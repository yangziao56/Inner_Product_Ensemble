#!/bin/bash
python IP-remove_food.py --dataset Food101N --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 0.01 --acc 80.63699641425859 >> Food101N.txt 2>&1
python IP-remove_food.py --dataset Food101N --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 0.01 --acc 80.63699641425859 --last_layer True >> Food101N_lastlayer.txt 2>&1


python IP-remove_food.py --dataset Food101N --repeat 3 --device cuda:1 --ensemble_size 15 --dropout_IP 0.01 --acc 80.63699641425859 >> Food101N_ensemble15.txt 2>&1
python IP-remove_food.py --dataset Food101N --repeat 3 --device cuda:1 --ensemble_size 10 --dropout_IP 0.01 --acc 80.63699641425859 >> Food101N_ensemble10.txt 2>&1
python IP-remove_food.py --dataset Food101N --repeat 3 --device cuda:1 --ensemble_size 5 --dropout_IP 0.01 --acc 80.63699641425859 >> Food101N_ensemble5.txt 2>&1
python IP-remove_food.py --dataset Food101N --repeat 3 --device cuda:1 --ensemble_size 1 --dropout_IP 0.01 --acc 80.63699641425859 >> Food101N_ensemble1.txt 2>&1

python IP-remove_food.py --dataset Food101N --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 0.1 --acc 80.63699641425859 >> Food101N_drop10%.txt 2>&1
python IP-remove_food.py --dataset Food101N --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 0.5 --acc 80.63699641425859 >> Food101N_drop50%.txt 2>&1
python IP-remove_food.py --dataset Food101N --repeat 3 --device cuda:1 --ensemble_size 20 --dropout_IP 1 --acc 80.63699641425859 >> Food101N_drop100s%.txt 2>&1