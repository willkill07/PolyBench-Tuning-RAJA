#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double *alpha, double *beta, double A[4000][4000], double u1[4000], double v1[4000], double u2[4000], double v2[4000], double w[4000], double x[4000], double y[4000], double z[4000]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;
  double fn = (double)n;
  for (i = 0; i < n; i++) {
    u1[i] = i;
    u2[i] = ((i + 1) / fn) / 2.0;
    v1[i] = ((i + 1) / fn) / 4.0;
    v2[i] = ((i + 1) / fn) / 6.0;
    y[i] = ((i + 1) / fn) / 8.0;
    z[i] = ((i + 1) / fn) / 9.0;
    x[i] = 0.0;
    w[i] = 0.0;
    for (j = 0; j < n; j++)
      A[i][j] = (double)(i * j % n) / n;
  }
}

static __attribute__ ((noinline)) void kernel_gemver(int n, double alpha, double beta, double A[4000][4000], double u1[4000], double v1[4000], double u2[4000], double v2[4000], double w[4000], double x[4000], double y[4000], double z[4000]) {
  int i, j;
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];
  for (i = 0; i < n; i++)
    x[i] = x[i] + z[i];
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      w[i] = w[i] + alpha * A[i][j] * x[j];
}

int main() {
  int n = 4000;
  double alpha;
  double beta;
  double(*A)[4000][4000];
  double(*u1)[4000];
  double(*v1)[4000];
  double(*u2)[4000];
  double(*v2)[4000];
  double(*w)[4000];
  double(*x)[4000];
  double(*y)[4000];
  double(*z)[4000];

  posix_memalign((void**)&A, 64, (4000) * (4000) * sizeof(double));
  posix_memalign((void**)&u1, 64, 4000 * sizeof(double));
  posix_memalign((void**)&v1, 64, 4000 * sizeof(double));
  posix_memalign((void**)&u2, 64, 4000 * sizeof(double));
  posix_memalign((void**)&v2, 64, 4000 * sizeof(double));
  posix_memalign((void**)&w, 64, 4000 * sizeof(double));
  posix_memalign((void**)&x, 64, 4000 * sizeof(double));
  posix_memalign((void**)&y, 64, 4000 * sizeof(double));
  posix_memalign((void**)&z, 64, 4000 * sizeof(double));

  init_array(n, &alpha, &beta, *A, *u1, *v1, *u2, *v2, *w, *x, *y, *z);

  DefaultTimer t;
  t.start();
  kernel_gemver(n, alpha, beta, *A, *u1, *v1, *u2, *v2, *w, *x, *y, *z);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("w", &(*w)[0], n);

  free((void *)A);
  free((void *)u1);
  free((void *)v1);
  free((void *)u2);
  free((void *)v2);
  free((void *)w);
  free((void *)x);
  free((void *)y);
  free((void *)z);
  return 0;
}
