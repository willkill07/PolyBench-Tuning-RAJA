#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double *float_n, double data[3000][2600]) {
  int i, j;
  *float_n = (double)m;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      data[i][j] = (double)(i * j) / n + i;
}

static __attribute__ ((noinline)) void kernel_correlation(int m, int n, double float_n, double data[3000][2600], double corr[2600][2600], double mean[2600], double stddev[2600]) {
  int i, j, k;
  double eps = 0.1;
  for (j = 0; j < m; j++) {
    mean[j] = 0.0;
    for (i = 0; i < n; i++)
      mean[j] += data[i][j];
    mean[j] /= float_n;
  }
  for (j = 0; j < m; j++) {
    stddev[j] = 0.0;
    for (i = 0; i < n; i++)
      stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
    stddev[j] /= float_n;
    stddev[j] = sqrt(stddev[j]);
    stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
  }
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      data[i][j] -= mean[j];
      data[i][j] /= sqrt(float_n) * stddev[j];
    }
  for (i = 0; i < m - 1; i++) {
    corr[i][i] = 1.0;
    for (j = i + 1; j < m; j++) {
      corr[i][j] = 0.0;
      for (k = 0; k < n; k++)
        corr[i][j] += (data[k][i] * data[k][j]);
      corr[j][i] = corr[i][j];
    }
  }
  corr[m - 1][m - 1] = 1.0;
}

int main() {
  int n = 3000;
  int m = 2600;
  double float_n;
  double(*data)[3000][2600];
  double(*corr)[2600][2600];
  double(*mean)[2600];
  double(*stddev)[2600];

  posix_memalign((void**)&data, 64, (3000) * (2600) * sizeof(double));
  posix_memalign((void**)&corr, 64, (2600) * (2600) * sizeof(double));
  posix_memalign((void**)&mean, 64, 2600 * sizeof(double));
  posix_memalign((void**)&stddev, 64, 2600 * sizeof(double));

  init_array(m, n, &float_n, *data);

  DefaultTimer t;
  t.start();
  kernel_correlation(m, n, float_n, *data, *corr, *mean, *stddev);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("corr", &(*corr)[0][0], m * m);

  free((void *)data);
  free((void *)corr);
  free((void *)mean);
  free((void *)stddev);
  return 0;
}
