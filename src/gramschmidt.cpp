#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int m, int n, double A[2000][2600], double R[2600][2600], double Q[2000][2600]) {
  int i, j;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      A[i][j] = (((double)((i * j) % m) / m) * 100) + 10;
      Q[i][j] = 0.0;
    }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      R[i][j] = 0.0;
}

static __attribute__ ((noinline)) void kernel_gramschmidt(int m, int n, double A[2000][2600], double R[2600][2600], double Q[2000][2600]) {
  RAJA::forall<Pol_Id_0_Size_1_Parent_null>(RAJA::RangeSegment{0, n}, [=] (int k) {
    RAJA::ReduceSum<typename Reduce<Pol_Id_1_Size_1_Parent_0>::type, double> nrm(0);
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{0, m}, [=] (int i) {
      nrm += A[i][k] * A[i][k];
    });
    R[k][k] = sqrt(nrm);
    RAJA::forall<Pol_Id_2_Size_1_Parent_0>(RAJA::RangeSegment{0, m}, [=] (int i) {
      Q[i][k] = A[i][k] / R[k][k];
    });
    RAJA::forall<Pol_Id_3_Size_1_Parent_0>(RAJA::RangeSegment{k + 1, n}, [=] (int j) {
      RAJA::ReduceSum<typename Reduce<Pol_Id_4_Size_1_Parent_3>::type, double> r(0);
      RAJA::forall<Pol_Id_4_Size_1_Parent_3>(RAJA::RangeSegment{0, m}, [=] (int i) {
        r += Q[i][k] * A[i][j];
      });
      double rr = r;
      RAJA::forall<Pol_Id_5_Size_1_Parent_3>(RAJA::RangeSegment{0, m}, [=] (int i) {
        A[i][j] = A[i][j] - Q[i][k] * rr;
      });
    });
  });
}

int main() {
  int m = 2000;
  int n = 2600;
  double(*A)[2000][2600];
  double(*R)[2600][2600];
  double(*Q)[2000][2600];

  posix_memalign((void**)&A, 64, (2000) * (2600) * sizeof(double));
  posix_memalign((void**)&R, 64, (2600) * (2600) * sizeof(double));
  posix_memalign((void**)&Q, 64, (2000) * (2600) * sizeof(double));

  init_array(m, n, *A, *R, *Q);

  DefaultTimer t;
  t.start();
  kernel_gramschmidt(m, n, *A, *R, *Q);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("A", &(*A)[0][0], m * n);
  dump("R", &(*R)[0][0], n * n);
  dump("Q", &(*Q)[0][0], m * n);

  free((void *)A);
  free((void *)R);
  free((void *)Q);
  return 0;
}
