#/!bin/bash

KERAS_BACKEND=theano THEANO_FLAGS=device=gpu0 python imgClassify.py 'test' $1 $2 $3 
