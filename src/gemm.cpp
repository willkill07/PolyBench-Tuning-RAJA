#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int ni, int nj, int nk, double *alpha, double *beta, double C[2000][2300], double A[2000][2600], double B[2600][2300]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (double)((i * j + 1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double)(i * (j + 1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double)(i * (j + 2) % nj) / nj;
}

static __attribute__ ((noinline)) void kernel_gemm(int ni, int nj, int nk, double alpha, double beta, double C[2000][2300], double A[2000][2600], double B[2600][2300]) {
  RAJA::forallN<Pol_Id_0_Size_2_Parent_null>(RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nj}, [=] (int i, int j) {
    C[i][j] *= beta;
  });
  RAJA::forallN<Pol_Id_1_Size_3_Parent_null>(RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nk}, RAJA::RangeSegment{0, nj}, [=] (int i, int k, int j) {
    C[i][j] += alpha * A[i][k] * B[k][j];
  });
}

int main() {
  int ni = 2000;
  int nj = 2300;
  int nk = 2600;
  double alpha;
  double beta;
  double(*C)[2000][2300];
  double(*A)[2000][2600];
  double(*B)[2600][2300];

  posix_memalign((void**)&C, 64, (2000) * (2300) * sizeof(double));
  posix_memalign((void**)&A, 64, (2000) * (2600) * sizeof(double));
  posix_memalign((void**)&B, 64, (2600) * (2300) * sizeof(double));

  init_array(ni, nj, nk, &alpha, &beta, *C, *A, *B);

  DefaultTimer t;
  t.start();
  kernel_gemm(ni, nj, nk, alpha, beta, *C, *A, *B);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("C", &(*C)[0][0], ni * nj);

  free((void *)C);
  free((void *)A);
  free((void *)B);
  return 0;
}
