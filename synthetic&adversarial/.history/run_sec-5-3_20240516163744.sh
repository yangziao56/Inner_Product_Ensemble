#!/bin/bash
# python SEC-5-3.py --dataset nlp --task relabel
# python SEC-5-3.py --dataset nlp --task remove
python SEC-5-3.py --dataset nlp --task reweight

# python SEC-5-3.py --dataset bank --task relabel
# python SEC-5-3.py --dataset bank --task remove
python SEC-5-3.py --dataset bank --task reweight

# python SEC-5-3.py --dataset celeba --task relabel
# python SEC-5-3.py --dataset celeba --task remove
python SEC-5-3.py --dataset celeba --task reweight
