/bin/hostname

cd ~/parallel-computing/Assignments/

module load impi-trial/5.0.1.035

make ex3

N=1000000000

echo "Running MPI Pi approximation with N=${N}">>times.txt
for np in 1 2 4 8 12 16 20 24 28 32 36 40
do
	echo "Number of processes: ${np}">>times.txt
	(mpirun -np ${np} ex3.x ${N})>>times.txt
done

exit