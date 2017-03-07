CXX := g++-5
CXXFLAGS := -O3 -march=native -Wall -Wextra -pedantic-errors -std=c++11

KERNELS := 2mm 3mm adi atax bicg cholesky correlation covariance deriche doitgen durbin fdtd-2d floyd-warshall gemm gemver gesummv gramschmidt heat-3d jacobi-1d jacobi-2d lu ludcmp mvt nussinov seidel-2d symm syr2k syrk trisolv trmm

RUNNERS := $(addprefix run-,$(KERNELS))

.PHONY: all runall clean

all : $(KERNELS)

runall : $(RUNNERS)

run-% : %
	./$<

clean :
	-$(RM) $(KERNELS)
