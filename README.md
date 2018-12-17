# Image Processing using OPENMP and MPI

## OpenMP

OpenMP is an API to write Multi-threaded programs. It has set of compiler directives and library routines in C, C++, and Fortran. \
You have a shared address space shared amongst all the threads. OpenMP uses teams of threads, and inside a parallel region, the work is distributed over the threads with a work sharing construct. 
Threads can access shared data, and they have some private data.
In this experiment, we have compared our execution time with serial and parallel with  Gaussian blur experiment.

### Steps to run OpenMPI

  1. make all
  2. ./opemmp 1 600 
      where 1 is the radius of the gaussian function and 600.bmp is the file name.
      
## OpenMPI

Message Passing Interface is a standardized message-passing system designed to achieve parallelism on a wide variety of parallel computers.
MPI runs on any data sharing model.
1. Distributed Memory
2. Shared Memory



