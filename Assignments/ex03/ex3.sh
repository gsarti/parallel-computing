/bin/hostname

cd ~/parallel-computing/Assignments/

module load impi-trial/5.0.1.035
module load openmpi/1.8.3/intel/14.0

make ex3

N=1000000000

echo "Running MPI Pi approximation with N=${N}"
for np in 1 2 4 8 12 16 20 24 28 32 36 40
do
	echo "Number of processes: ${np}"
	mpirun -np ${np} ex3.x ${N}
done

exit
