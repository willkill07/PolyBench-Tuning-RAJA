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
  RAJA::forallN<Pol_Id_0_Size_2_Parent_Nil>(RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
    A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
  });
  RAJA::forallN<Pol_Id_1_Size_2_Parent_Nil>(RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
    x[i] = x[i] + beta * A[j][i] * y[j];
  });
  RAJA::forall<Pol_Id_2_Size_1_Parent_Nil>(RAJA::RangeSegment{0, n}, [=] (int i) {
    x[i] = x[i] + z[i];
  });
  RAJA::forallN<Pol_Id_3_Size_2_Parent_Nil>(RAJA::RangeSegment{0, n}, RAJA::RangeSegment{0, n}, [=] (int i, int j) {
    w[i] = w[i] + alpha * A[i][j] * x[j];
  });
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
