#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double r[4000]) {
  int i;
  for (i = 0; i < n; i++) {
    r[i] = (n + 1 - i);
  }
}

static __attribute__ ((noinline)) void kernel_durbin(int n, double r[4000], double y[4000]) {
  double z[4000];
  double alpha;
  double beta;
  double sum;
  int i, k;
  y[0] = -r[0];
  beta = 1.0;
  alpha = -r[0];
  for (k = 1; k < n; k++) {
    beta = (1 - alpha * alpha) * beta;
    sum = 0.0;
    for (i = 0; i < k; i++) {
      sum += r[k - i - 1] * y[i];
    }
    alpha = -(r[k] + sum) / beta;
    for (i = 0; i < k; i++) {
      z[i] = y[i] + alpha * y[k - i - 1];
    }
    for (i = 0; i < k; i++) {
      y[i] = z[i];
    }
    y[k] = alpha;
  }
}

int main() {
  int n = 4000;
  double(*r)[4000];
  double(*y)[4000];

  posix_memalign((void**)&r, 64, 4000 * sizeof(double));
  posix_memalign((void**)&y, 64, 4000 * sizeof(double));

  init_array(n, *r);

  DefaultTimer t;
  t.start();
  kernel_durbin(n, *r, *y);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("y", &(*y)[0], n);

  free((void *)r);
  free((void *)y);
  return 0;
}
