#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int ni, int nj, int nk, int nl, int nm, double A[1600][2000], double B[2000][1800], double C[1800][2400], double D[2400][2200]) {
  int i, j;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (double)((i * j + 1) % ni) / (5 * ni);
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (double)((i * (j + 1) + 2) % nj) / (5 * nj);
  for (i = 0; i < nj; i++)
    for (j = 0; j < nm; j++)
      C[i][j] = (double)(i * (j + 3) % nl) / (5 * nl);
  for (i = 0; i < nm; i++)
    for (j = 0; j < nl; j++)
      D[i][j] = (double)((i * (j + 2) + 2) % nk) / (5 * nk);
}

static __attribute__ ((noinline)) void kernel_3mm(int ni, int nj, int nk, int nl, int nm, double E[1600][1800], double A[1600][2000], double B[2000][1800], double F[1800][2200], double C[1800][2400], double D[2400][2200], double G[1600][2200]) {
  RAJA::forallN<Pol_Id_0_Size_2_Parent_Nil> (RAJA::RangeSegment{0, ni}, RAJA::RangeSegment{0, nj}, [=] (int i, int j) {
    RAJA::ReduceSum<Pol_Id_1_Size_1_Parent_0, double> e(0);
    RAJA::forall<Pol_Id_1_Size_1_Parent_0> (RAJA::RangeSegment{0, nk}, [=] (int k) {
      e += A[i][k] * B[k][j];
    });
    E[i][j] = e;
  });
  RAJA::forallN<Pol_Id_2_Size_2_Parent_Nil> (RAJA::RangeSegment{0, nj}, RAJA::RangeSegment{0, nl}, [=] (int i, int j) {
    RAJA::ReduceSum<Pol_Id_1_Size_1_Parent_0, double> f(0);
    RAJA::forall<Pol_Id_3_Size_1_Parent_2> (RAJA::RangeSegment{0, nm}, [=] (int k) {
      f += C[i][k] * D[k][j];
    });
    F[i][j] = f;
  });
  RAJA::forallN<Pol_Id_4_Size_2_Parent_Nil> (RAJA::RangeSegment{0, nj}, RAJA::RangeSegment{0, nl}, [=] (int i, int j) {
    RAJA::ReduceSum<Pol_Id_1_Size_1_Parent_0, double> g(0);
    RAJA::forall<Pol_Id_5_Size_1_Parent_4> (RAJA::RangeSegment{0, nm}, [=] (int k) {
      g += E[i][k] * F[k][j];
    });
    G[i][j] = g;
  });
}

int main() {
  int ni = 1600;
  int nj = 1800;
  int nk = 2000;
  int nl = 2200;
  int nm = 2400;
  double(*E)[1600][1800];
  double(*A)[1600][2000];
  double(*B)[2000][1800];
  double(*F)[1800][2200];
  double(*C)[1800][2400];
  double(*D)[2400][2200];
  double(*G)[1600][2200];

  posix_memalign((void**)&E, 64, (1600) * (1800) * sizeof(double));
  posix_memalign((void**)&A, 64, (1600) * (2000) * sizeof(double));
  posix_memalign((void**)&B, 64, (2000) * (1800) * sizeof(double));
  posix_memalign((void**)&F, 64, (1800) * (2200) * sizeof(double));
  posix_memalign((void**)&C, 64, (1800) * (2400) * sizeof(double));
  posix_memalign((void**)&D, 64, (2400) * (2200) * sizeof(double));
  posix_memalign((void**)&G, 64, (1600) * (2200) * sizeof(double));

  init_array(ni, nj, nk, nl, nm, *A, *B, *C, *D);
  DefaultTimer t;
  t.start();
  kernel_3mm(ni, nj, nk, nl, nm, *E, *A, *B, *F, *C, *D, *G);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("G", &(*G)[0][0], ni * nl);

  free((void *)E);
  free((void *)A);
  free((void *)B);
  free((void *)F);
  free((void *)C);
  free((void *)D);
  free((void *)G);
  return 0;
}
