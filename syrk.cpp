#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, int m, double *alpha, double *beta, double C[2600][2600], double A[2600][2000]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (double)((i * j + 1) % n) / n;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      C[i][j] = (double)((i * j + 2) % m) / m;
}

static __attribute__ ((noinline)) void kernel_syrk(int n, int m, double alpha, double beta, double C[2600][2600], double A[2600][2000]) {
  int i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < m; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }
}

int main() {
  int n = 2600;
  int m = 2000;
  double alpha;
  double beta;
  double(*C)[2600][2600];
  double(*A)[2600][2000];

  posix_memalign((void**)&C, 64, (2600) * (2600) * sizeof(double));
  posix_memalign((void**)&A, 64, (2600) * (2000) * sizeof(double));

  init_array(n, m, &alpha, &beta, *C, *A);

  DefaultTimer t;
  t.start();
  kernel_syrk(n, m, alpha, beta, *C, *A);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("C", &(*C)[0], n);

  free((void *)C);
  free((void *)A);
  return 0;
}
