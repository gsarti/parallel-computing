/bin/hostname

cd ~/parallel-computing/Assignments/

make

for thread in 1 2 4 8 16 20
do
    export OMP_NUM_THREADS=${thread}
    echo "−−−−−−−−−−−−−−−−−−"
    echo "Run with ${thread} thread(s)"
    echo "−−−−−−−−−−−−−−−−−−"
    ./ex1.x 1000000000
done

exit
