#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double L[4000][4000], double x[4000], double b[4000]) {
  int i, j;
  for (i = 0; i < n; i++) {
    x[i] = -999;
    b[i] = i;
    for (j = 0; j <= i; j++)
      L[i][j] = (double)(i + n - j + 1) * 2 / n;
  }
}

static __attribute__ ((noinline)) void kernel_trisolv(int n, double L[4000][4000], double x[4000], double b[4000]) {
  int i, j;
  for (i = 0; i < n; i++) {
    x[i] = b[i];
    for (j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}

int main() {
  int n = 4000;
  double(*L)[4000][4000];
  double(*x)[4000];
  double(*b)[4000];

  posix_memalign((void**)&L, 64, (4000) * (4000) * sizeof(double));
  posix_memalign((void**)&x, 64, 4000 * sizeof(double));
  posix_memalign((void**)&b, 64, 4000 * sizeof(double));

  init_array(n, *L, *x, *b);

  DefaultTimer t;
  t.start();
  kernel_trisolv(n, *L, *x, *b);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("x", &(*x)[0], n);

  free((void *)L);
  free((void *)x);
  free((void *)b);
  return 0;
}
