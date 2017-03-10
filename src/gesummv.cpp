#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double *alpha, double *beta, double A[2800][2800], double B[2800][2800], double x[2800]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;

  if(load_init("A", A[0], n * n) &&
     load_init("B", B[0], n * n) &&
     load_init("x", x, n))
    return;

  for (i = 0; i < n; i++) {
    x[i] = (double)(i % n) / n;
    for (j = 0; j < n; j++) {
      A[i][j] = (double)((i * j + 1) % n) / n;
      B[i][j] = (double)((i * j + 2) % n) / n;
    }
  }

  dump_init("A", A[0], n * n);
  dump_init("B", B[0], n * n);
  dump_init("x", x, n);
}

static __attribute__ ((noinline)) void kernel_gesummv(int n, double alpha, double beta, double A[2800][2800], double B[2800][2800], double tmp[2800], double x[2800], double y[2800]) {
  RAJA::forall<Pol_Id_0_Size_1_Parent_null>(RAJA::RangeSegment{0, n}, [=] (int i) {
    RAJA::ReduceSum<typename Reduce<Pol_Id_1_Size_1_Parent_0>::type, double> t(0), yy(0);
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{0, n}, [=] (int j) {
      t += A[i][j] * x[j];
      yy += B[i][j] * x[j];
    });
    y[i] = alpha * t + beta * yy;
  });
}

int main() {
  int n = 2800;
  double alpha;
  double beta;
  double(*A)[2800][2800];
  double(*B)[2800][2800];
  double(*tmp)[2800];
  double(*x)[2800];
  double(*y)[2800];
  posix_memalign((void**)&A, 64, (2800) * (2800) * sizeof(double));
  posix_memalign((void**)&B, 64, (2800) * (2800) * sizeof(double));
  posix_memalign((void**)&tmp, 64, 2800 * sizeof(double));
  posix_memalign((void**)&x, 64, 2800 * sizeof(double));
  posix_memalign((void**)&y, 64, 2800 * sizeof(double));

  init_array(n, &alpha, &beta, *A, *B, *x);

  DefaultTimer t;
  t.start();
  kernel_gesummv(n, alpha, beta, *A, *B, *tmp, *x, *y);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("y", &(*y)[0], n);

  free((void *)A);
  free((void *)B);
  free((void *)tmp);
  free((void *)x);
  free((void *)y);
  return 0;
}
