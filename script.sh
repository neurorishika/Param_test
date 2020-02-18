#!/bin/sh 

for f in scripts/*.sh
do
	sbatch -W $f
done
wait
echo "Completed"
