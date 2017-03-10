#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, int m, double *alpha, double *beta, double C[2600][2600], double A[2600][2000], double B[2600][2000]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;

  if(load_init("C", C[0], n * n) &&
     load_init("B", B[0], m * n) &&
     load_init("A", A[0], m * n))
    return;

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      A[i][j] = (double)((i * j + 1) % n) / n;
      B[i][j] = (double)((i * j + 2) % m) / m;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (double)((i * j + 3) % n) / m;
    }

  dump_init("C", C[0], n * n);
  dump_init("B", B[0], m * n);
  dump_init("A", A[0], m * n);
}

static __attribute__ ((noinline)) void kernel_syr2k(int n, int m, double alpha, double beta, double C[2600][2600], double A[2600][2000], double B[2600][2000]) {
  RAJA::forallN<Pol_Id_0_Size_2_Parent_null>(RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
    if (j <= i)
      C[i][j] *= beta;
  });
  RAJA::forallN<Pol_Id_1_Size_3_Parent_null>(RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, m}, RAJA::RangeSegment{0, n}, [=] (int i, int k, int j) {
    if (j <= i)
      C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];
  });
}

int main() {
  int n = 2600;
  int m = 2000;
  double alpha;
  double beta;
  double(*C)[2600][2600];
  double(*A)[2600][2000];
  double(*B)[2600][2000];

  posix_memalign((void**)&C, 64, (2600) * (2600) * sizeof(double));
  posix_memalign((void**)&A, 64, (2600) * (2000) * sizeof(double));
  posix_memalign((void**)&B, 64, (2600) * (2000) * sizeof(double));

  init_array(n, m, &alpha, &beta, *C, *A, *B);

  DefaultTimer t;
  t.start();
  kernel_syr2k(n, m, alpha, beta, *C, *A, *B);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("C", &(*C)[0][0], n * n);

  free((void *)C);
  free((void *)A);
  free((void *)B);
  return 0;
}
