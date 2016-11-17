#/!bin/bash

KERAS_BACKEND=theano THEANO_FLAGS=device=gpu0 python imgClassify.py 'train' $1 $2 'T' 'F' 'T'
