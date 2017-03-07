#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double A[4000][4000], double b[4000], double x[4000], double y[4000]) {
  int i, j;
  double fn = (double)n;
  for (i = 0; i < n; i++) {
    x[i] = 0;
    y[i] = 0;
    b[i] = (i + 1) / fn / 2.0 + 4;
  }
  for (i = 0; i < n; i++) {
    for (j = 0; j <= i; j++)
      A[i][j] = (double)(-j % n) / n + 1;
    for (j = i + 1; j < n; j++) {
      A[i][j] = 0;
    }
    A[i][i] = 1;
  }
  int r, s, t;
  double(*B)[4000][4000];
  posix_memalign((void**)&B, 64, (4000) * (4000) * sizeof(double));
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      (*B)[r][s] = 0;
  for (t = 0; t < n; ++t)
    for (r = 0; r < n; ++r)
      for (s = 0; s < n; ++s)
        (*B)[r][s] += A[r][t] * A[s][t];
  for (r = 0; r < n; ++r)
    for (s = 0; s < n; ++s)
      A[r][s] = (*B)[r][s];
  free((void *)B);
}

static __attribute__ ((noinline)) void kernel_ludcmp(int n, double A[4000][4000], double b[4000], double x[4000], double y[4000]) {
  int i, j, k;
  double w;
  for (i = 0; i < n; i++) {
    for (j = 0; j < i; j++) {
      w = A[i][j];
      for (k = 0; k < j; k++) {
        w -= A[i][k] * A[k][j];
      }
      A[i][j] = w / A[j][j];
    }
    for (j = i; j < n; j++) {
      w = A[i][j];
      for (k = 0; k < i; k++) {
        w -= A[i][k] * A[k][j];
      }
      A[i][j] = w;
    }
  }
  for (i = 0; i < n; i++) {
    w = b[i];
    for (j = 0; j < i; j++)
      w -= A[i][j] * y[j];
    y[i] = w;
  }
  for (i = n - 1; i >= 0; i--) {
    w = y[i];
    for (j = i + 1; j < n; j++)
      w -= A[i][j] * x[j];
    x[i] = w / A[i][i];
  }
}

int main() {
  int n = 4000;
  double(*A)[4000][4000];
  double(*b)[4000];
  double(*x)[4000];
  double(*y)[4000];

  posix_memalign((void**)&A, 64, (4000) * (4000) * sizeof(double));
  posix_memalign((void**)&b, 64, 4000 * sizeof(double));
  posix_memalign((void**)&x, 64, 4000 * sizeof(double));
  posix_memalign((void**)&y, 64, 4000 * sizeof(double));

  init_array(n, *A, *b, *x, *y);

  DefaultTimer t;
  t.start();
  kernel_ludcmp(n, *A, *b, *x, *y);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("x", &(*x)[0], n);

  free((void *)A);
  free((void *)b);
  free((void *)x);
  free((void *)y);
  return 0;
}
