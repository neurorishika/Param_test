#!/bin/sh 

for f in scripts/p1.50/*.sh
do
	sbatch -W $f
done
wait
echo "Completed"
