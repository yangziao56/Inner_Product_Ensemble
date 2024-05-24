#!/bin/bash
python IP-remove_animal.py --repeat 5 --device cuda:1 --ensemble_size 5 --dropout_IP 0.01  >> result_animal/animal.txt 2>&1
python IP-remove_animal.py --repeat 5 --device cuda:1 --ensemble_size 5 --dropout_IP 0.01 --last_layer True >> result_animal/animal_last_layer.txt 2>&1

python IP-remove_animal.py --repeat 5 --device cuda:1 --ensemble_size 15 --dropout_IP 0.01 >> result_animal/animal_ensemble15.txt 2>&1
python IP-remove_animal.py --repeat 5 --device cuda:1 --ensemble_size 10 --dropout_IP 0.01 >> result_animal/animal_ensemble10.txt 2>&1
#python IP-remove_animal.py --repeat 5 --device cuda:1 --ensemble_size 5 --dropout_IP 0.01 >> result_animal/animal_ensemble5.txt 2>&1
python IP-remove_animal.py --repeat 5 --device cuda:1 --ensemble_size 1 --dropout_IP 0.01 >> result_animal/animal_ensemble1.txt 2>&1

python IP-remove_animal.py --repeat 5 --device cuda:1 --ensemble_size 5 --dropout_IP 0.1 >> result_animal/animal_drop10%.txt 2>&1
python IP-remove_animal.py --repeat 5 --device cuda:1 --ensemble_size 5 --dropout_IP 0.5 >> result_animal/animal_drop50%.txt 2>&1
python IP-remove_animal.py --repeat 5 --device cuda:1 --ensemble_size 5 --dropout_IP 1 >> result_animal/animal_drop100%.txt 2>&1