MPICC = mpiicc
FLAGS_OPENMP = -qopenmp
FLAGS_DEBUG = -g -00 
FLAGS_FAST = -O3 -xAVX -fno-alias

SRCDIR=.


all: clean release 

debug: 
	$(MPICC) $(FLAGS_DEBUG) $(FLAGS_OPENMP) -c $(SRCDIR)/utilities.c -o utilities.o
	$(MPICC) $(FLAGS_DEBUG) $(FLAGS_OPENMP) -c $(SRCDIR)/jacobi.c -o jacobi.o
	$(MPICC) $(FLAGS_DEBUG) $(FLAGS_OPENMP) -DREAD_INPUT -c $(SRCDIR)/main.c -o main.o
	$(MPICC) $(FLAGS_DEBUG) $(FLAGS_OPENMP) utilities.o jacobi.o main.o -o jacobi.exe -lm

release: 
	$(MPICC) $(FLAGS_OPENMP) $(FLAGS_FAST) -c $(SRCDIR)/utilities.c -o utilities.o
	$(MPICC) $(FLAGS_OPENMP) $(FLAGS_FAST) -c $(SRCDIR)/jacobi.c -o jacobi.o
	$(MPICC) $(FLAGS_OPENMP) $(FLAGS_FAST) -DREAD_INPUT -c $(SRCDIR)/main.c -o main.o
	$(MPICC) $(FLAGS_OPENMP) $(FLAGS_FAST) utilities.o jacobi.o main.o -o jacobi.exe -lm


clean:
	@rm -rf *.exe core* *.o *.mod a.out 2> /dev/null
