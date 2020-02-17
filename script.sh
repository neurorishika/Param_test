#!/bin/sh 

for f in scripts/*.sh
do
	sbatch $f
done
