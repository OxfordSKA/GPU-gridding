CC = g++
CFLAGS = -I. -O2 -g -std=c++11 
DEPS = oskar_grid_wproj.hpp Makefile gpu_support.hpp utils.hpp bmp_support.hpp

NVCC := nvcc    

# REMARK: Removed fast_math since it was slighlty slower. No idea why. SHuld not matter since most of the complex math calls are using double
# REMARK: Added -Wno-deprecated-declarations to remove CUDA warnings. TODO: port to new safer api
NVCCFLAGS := -std=c++11 -arch=sm_60 -Xptxas -v -O2 -g -restrict -use_fast_math -D_FORCE_INLINES -D_DEBUG -lineinfo
#NVCCFLAGS :=  -std=c++11 -arch=sm_60 -Xptxas -v -O2 -g -restrict -D_FORCE_INLINES -D_DEBUG

OBJ := ob.main.cpp.o ob.oskar.cu.o ob.gpu_support.cu.o ob.bmpfile.c.o
# main_orig is the original version without my tuning
OBJ_ORIG := ob.main.cpp.o ob.oskar_orig.cu.o ob.gpu_support.cu.o ob.bmpfile.c.o
LIBS = -lm


.PHONEY: all

all : main 


main : $(OBJ)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(LIBS)

main_orig : $(OBJ_ORIG)
	$(NVCC) -o $@ $^ $(NVCCFLAGS) $(LIBS)


ob.%.cpp.o : %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

ob.%.c.o : %.c $(DEPS)
	gcc -c -o $@ $< -I. -O3 -g

ob.%.cu.o : %.cu $(DEPS)
	$(NVCC) --ptx $< $(NVCCFLAGS)
	$(NVCC) -c -o $@ $< $(NVCCFLAGS)

run : main
	./$< ../../../data/oskar_grid_wproj_f_INPUT.dat ../../../data/oskar_grid_wproj_f_OUTPUT_float.dat  
run1 : main
	./$< ../../../data/oskar_grid_wproj_f_INPUT_EL30-EL56.dat ../../../data/oskar_grid_wproj_f_OUTPUT_EL30-EL56.dat 
run2 : main
	./$< ../../../data/oskar_grid_wproj_f_INPUT_EL56-EL82.dat ../../../data/oskar_grid_wproj_f_OUTPUT_EL56-EL82.dat 
run3 : main
	./$< ../../../data/oskar_grid_wproj_f_INPUT_EL82-EL70.dat ../../../data/oskar_grid_wproj_f_OUTPUT_EL82-EL70.dat 

clean:
	rm -f ob.*.o main main_orig

tar:
	tar czf ../oskar-tuned.tar.gz *.c *.cu *.cu_* *.cpp *.h *.hpp README LICENSE Makefile
