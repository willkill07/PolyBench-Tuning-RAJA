#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double A[200][200][200], double B[200][200][200]) {
  int i, j, k;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (double)(i + j + (n - k)) * 10 / (n);
}


static __attribute__ ((noinline)) void kernel_heat_3d(int tsteps, int n, double A[200][200][200], double B[200][200][200]) {
  int t, i, j, k;
  for (t = 1; t <= tsteps; t++) {
    for (i = 1; i < n - 1; i++) {
      for (j = 1; j < n - 1; j++) {
        for (k = 1; k < n - 1; k++) {
          B[i][j][k] = 0+ 0.125 * (A[i + 1][j][k] - 2.0 * A[i][j][k] + A[i - 1][j][k])+ 0.125 * (A[i][j + 1][k] - 2.0 * A[i][j][k] + A[i][j - 1][k])+ 0.125 * (A[i][j][k + 1] - 2.0 * A[i][j][k] + A[i][j][k - 1]) + A[i][j][k];
        }
      }
    }
    for (i = 1; i < n - 1; i++) {
      for (j = 1; j < n - 1; j++) {
        for (k = 1; k < n - 1; k++) {
          A[i][j][k] = 0+ 0.125 * (B[i + 1][j][k] - 2.0 * B[i][j][k] + B[i - 1][j][k])+ 0.125 * (B[i][j + 1][k] - 2.0 * B[i][j][k] + B[i][j - 1][k])+ 0.125 * (B[i][j][k + 1] - 2.0 * B[i][j][k] + B[i][j][k - 1]) + B[i][j][k];
        }
      }
    }
  }
}

int main() {
  int n = 200;
  int tsteps = 1000;
  double(*A)[200][200][200];
  double(*B)[200][200][200];

  posix_memalign((void**)&A, 64, (200) * (200) * (200) * sizeof(double));
  posix_memalign((void**)&B, 64, (200) * (200) * (200) * sizeof(double));

  init_array(n, *A, *B);

  DefaultTimer t;
  t.start();
  kernel_heat_3d(tsteps, n, *A, *B);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("A", &(*A)[0][0][0], n * n * n);

  free((void *)A);
  return 0;
}
