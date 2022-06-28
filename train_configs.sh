#!/bin/bash

source $1/bin/activate;

for filename in ./configs/*.json; 
	do python3 main.py 0 $(basename "$filename");
done
    
