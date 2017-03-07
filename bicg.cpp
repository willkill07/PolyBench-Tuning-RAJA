#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double A[2200][1800], double r[2200], double p[1800]) {
  int i, j;
  for (i = 0; i < m; i++)
    p[i] = (double)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (double)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (double)(i * (j + 1) % n) / n;
  }
}

static __attribute__ ((noinline)) void kernel_bicg(int m, int n, double A[2200][1800], double s[1800], double q[2200], double p[1800], double r[2200]) {
  int i, j;
  for (i = 0; i < m; i++)
    s[i] = 0;
  for (i = 0; i < n; i++) {
    q[i] = 0.0;
    for (j = 0; j < m; j++) {
      s[j] = s[j] + r[i] * A[i][j];
      q[i] = q[i] + A[i][j] * p[j];
    }
  }
}

int main() {
  int n = 2200;
  int m = 1800;
  double(*A)[2200][1800];
  double(*s)[1800];
  double(*q)[2200];
  double(*p)[1800];
  double(*r)[2200];

  posix_memalign((void**)&A, 64, (2200) * (1800) * sizeof(double));
  posix_memalign((void**)&s, 64, 1800 * sizeof(double));
  posix_memalign((void**)&q, 64, 2200 * sizeof(double));
  posix_memalign((void**)&p, 64, 1800 * sizeof(double));
  posix_memalign((void**)&r, 64, 2200 * sizeof(double));

  init_array(m, n, *A, *r, *p);

  DefaultTimer t;
  t.start();
  kernel_bicg(m, n, *A, *s, *q, *p, *r);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("s", &(*s)[0], m);
  dump("q", &(*q)[0], n);

  free((void *)A);
  free((void *)s);
  free((void *)q);
  free((void *)p);
  free((void *)r);
  return 0;
}
