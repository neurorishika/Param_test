#!/bin/sh 

for f in scripts/p2.25/*.sh
do
	sbatch -W $f
done
wait
echo "Completed"
