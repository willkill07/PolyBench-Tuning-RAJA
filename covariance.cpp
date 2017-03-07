#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double *float_n, double data[3000][2600]) {
  int i, j;
  *float_n = (double)n;
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      data[i][j] = ((double)i * j) / m;
}

static __attribute__ ((noinline)) void kernel_covariance(int m, int n, double float_n, double data[3000][2600], double cov[2600][2600], double mean[2600]) {
  int i, j, k;
  for (j = 0; j < m; j++) {
    mean[j] = 0.0;
    for (i = 0; i < n; i++) {
      mean[j] += data[i][j];
    }
    mean[j] /= float_n;
  }
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      data[i][j] -= mean[j];
    }
  }
  for (i = 0; i < m; i++) {
    for (j = i; j < m; j++) {
      cov[i][j] = 0.0;
      for (k = 0; k < n; k++) {
        cov[i][j] += data[k][i] * data[k][j];
      }
      cov[i][j] /= (float_n - 1.0);
      cov[j][i] = cov[i][j];
    }
  }
}

int main() {
  int n = 3000;
  int m = 2600;
  double float_n;
  double(*data)[3000][2600];
  double(*cov)[2600][2600];
  double(*mean)[2600];

  posix_memalign((void**)&data, 64, (3000) * (2600) * sizeof(double));
  posix_memalign((void**)&cov, 64, (2600) * (2600) * sizeof(double));
  posix_memalign((void**)&mean, 64, 2600 * sizeof(double));

  init_array(m, n, &float_n, *data);

  DefaultTimer t;
  t.start();
  kernel_covariance(m, n, float_n, *data, *cov, *mean);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("cov", &(*cov)[0][0], m * m);

  free((void *)data);
  free((void *)cov);
  free((void *)mean);
  return 0;
}
