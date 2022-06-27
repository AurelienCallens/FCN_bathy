#!/bin/bash

source tf_gpu/bin/activate;

for filename in ./configs/*.json; 
	do python3 test_bash.py $(basename "$filename") ;
done
    
