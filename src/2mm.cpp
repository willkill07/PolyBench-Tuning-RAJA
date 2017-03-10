#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int ni, int nj, int nk, int nl, double *alpha, double *beta, double A[1600][2200], double B[2200][1800], double C[1800][2400], double D[1600][2400]) {
  int i, j;
  *alpha = 1.5;
  *beta = 1.2;

  if (load_init ("A", A[0], ni * nk) &&
      load_init ("B", B[0], nk * nj) &&
      load_init ("C", C[0], nj * nl) &&
      load_init ("D", D[0], nl * ni))
    return;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double)((i * j + 1) % ni) / ni;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double)(i * (j + 1) % nj) / nj;
  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C[i][j] = (double)((i * (j + 3) + 1) % nl) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (double)(i * (j + 2) % nk) / nk;

  dump_init ("A", A[0], ni * nk);
  dump_init ("B", B[0], nk * nj);
  dump_init ("C", C[0], nj * nl);
  dump_init ("D", D[0], nl * ni);
}

static __attribute__((noinline)) void kernel_2mm(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[1600][1800], double A[1600][2200], double B[2200][1800], double C[1800][2400], double D[1600][2400]) {
  RAJA::forallN<Pol_Id_0_Size_2_Parent_null> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nj}, [=] (int i, int j) {
    RAJA::ReduceSum<typename Reduce<Pol_Id_1_Size_1_Parent_0>::type, double> t(0);
    RAJA::forall<Pol_Id_1_Size_1_Parent_0> (RAJA::RangeSegment{0, nk}, [=] (int k) {
      t += alpha * A[i][k] * B[k][j];
    });
    tmp[i][j] = t;
  });
  RAJA::forallN<Pol_Id_2_Size_2_Parent_null> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nl}, [=] (int i, int j) {
    RAJA::ReduceSum<typename Reduce<Pol_Id_3_Size_1_Parent_2>::type, double> d(0);
    RAJA::forall<Pol_Id_3_Size_1_Parent_2> (RAJA::RangeSegment{0, nj}, [=] (int k) {
      d += tmp[i][k] * C[k][j];
    });
    D[i][j] = D[i][j] * beta + d;;
  });
}

int main() {
  int ni = 1600;
  int nj = 1800;
  int nk = 2200;
  int nl = 2400;
  double alpha;
  double beta;
  double(*tmp)[1600][1800];
  double(*A)[1600][2200];
  double(*B)[2200][1800];
  double(*C)[1800][2400];
  double(*D)[1600][2400];

  posix_memalign((void**)&tmp, 64, (1600) * (1800) * sizeof(double));
  posix_memalign((void**)&A, 64, (1600) * (2200) * sizeof(double));
  posix_memalign((void**)&B, 64, (2200) * (1800) * sizeof(double));
  posix_memalign((void**)&C, 64, (1800) * (2400) * sizeof(double));
  posix_memalign((void**)&D, 64, (1600) * (2400) * sizeof(double));

  init_array(ni, nj, nk, nl, &alpha, &beta, *A, *B, *C, *D);

  DefaultTimer t;
  t.start();
  kernel_2mm(ni, nj, nk, nl, alpha, beta, *tmp, *A, *B, *C, *D);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("D", &(*D)[0][0], ni * nl);

  free((void *)tmp);
  free((void *)A);
  free((void *)B);
  free((void *)C);
  free((void *)D);
  return 0;
}
