#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double *alpha, double A[2000][2000], double B[2000][2600]) {
  int i, j;
  *alpha = 1.5;
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++) {
      A[i][j] = (double)((i + j) % m) / m;
    }
    A[i][i] = 1.0;
    for (j = 0; j < n; j++) {
      B[i][j] = (double)((n + (i - j)) % n) / n;
    }
  }
}

static __attribute__ ((noinline)) void kernel_trmm(int m, int n, double alpha, double A[2000][2000], double B[2000][2600]) {
  int i, j, k;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      for (k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
}

int main() {
  int m = 2000;
  int n = 2600;
  double alpha;
  double(*A)[2000][2000];
  double(*B)[2000][2600];

  posix_memalign((void**)&A, 64, (2000) * (2000) * sizeof(double));
  posix_memalign((void**)&B, 64, (2000) * (2600) * sizeof(double));

  init_array(m, n, &alpha, *A, *B);

  DefaultTimer t;
  t.start();
  kernel_trmm(m, n, alpha, *A, *B);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("B", &(*B)[0][0], m * n);

  free((void *)A);
  free((void *)B);
  return 0;
}
