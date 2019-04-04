/bin/hostname

cd ~/parallel-computing/Assignments/

make ex2

export OMP_NUM_THREADS=10
echo "−−−−−−−−−−−−−−−−−−"
echo "Run ex2 with 10 threads"
echo "−−−−−−−−−−−−−−−−−−"
./ex2.x

exit