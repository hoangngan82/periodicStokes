CPP = nvcc
CPPFLAGS13 = -arch=sm_13 -ftz=true -prec-div=false -prec-sqrt=false
CPPFLAGS20 = -arch=sm_20 -ftz=false -prec-div=true -prec-sqrt=true
CPPFLAGS30 = -arch=sm_30 -ftz=false -prec-div=true -prec-sqrt=true --ptxas-options="-v"
CPPFLAGS = $(CPPFLAGS30) -O3
GSLFLAGS = `pkg-config --cflags --libs gsl` 
thesis: periodicStokesProjects.cu periodicStokesProjects.h 
	$(CPP) $(CPPFLAGS) $< -o /tmp/hoangngan/periodicStokes

matrix: matrixtest.cu  matrix.h
	$(CPP) $(CPPFLAGS) $< -o /tmp/hoangngan/matrixtest

Liron: testLironPaper.cu  
	$(CPP) $(CPPFLAGS) $(GSLFLAGS) $< -o /tmp/hoangngan/testLironPaper

optimal: optimalSplitting.cu periodicStokes.h
	$(CPP) $(CPPFLAGS) $< -lboost_system -lboost_timer -o /tmp/hoangngan/optimal

GPUtime: CPU_GPU_time.cu periodicStokes.h
	$(CPP) $(CPPFLAGS) $< -lboost_system -lboost_timer -o /tmp/hoangngan/GPUtime

runmatrix: 
	optirun --no-xorg cuda-memcheck /tmp/hoangngan/matrixtest

clean:
	rm -f *.o
