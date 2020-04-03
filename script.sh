#!/bin/sh 

for f in scripts/p1.75/*.sh
do
	sbatch -W $f
done
wait
echo "Completed"
