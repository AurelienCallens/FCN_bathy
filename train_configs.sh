#!/bin/bash

source env/bin/activate;

for filename in ./configs/*.json; 
	do python3 train.py 1 $(basename "$filename");
done
    
