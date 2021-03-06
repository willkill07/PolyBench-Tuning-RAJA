#include "polybench.hpp"

static __attribute__((noinline)) void init_array(int n, double u[2000][2000]) {
  int i, j;

  if (load_init ("u", u[0], n * n))
    return;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      u[i][j] = (double)(i + n - j) / n;
    }

  dump_init ("u", u[0], n * n);
}

static __attribute__ ((noinline)) void kernel_adi(int tsteps, int n, double u[2000][2000], double v[2000][2000], double p[2000][2000], double q[2000][2000]) {
  double DX, DY, DT;
  double B1, B2;
  double mul1, mul2;
  double a, b, c, d, e, f;
  DX = 1.0 / (double)n;
  DY = 1.0 / (double)n;
  DT = 1.0 / (double)tsteps;
  B1 = 2.0;
  B2 = 1.0;
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);
  a = -mul1 / 2.0;
  b = 1.0 + mul1;
  c = a;
  d = -mul2 / 2.0;
  e = 1.0 + mul2;
  f = d;
  for (int t = 1; t <= tsteps; t++) {
    RAJA::forall<Pol_Id_0_Size_1_Parent_null>(RAJA::RangeSegment{1, n - 1}, [=] (int i) {
      v[0][i] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = v[0][i];
      RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{1, n - 1}, [=] (int j) {
        p[i][j] = -c / (a * p[i][j - 1] + b);
        q[i][j] = (-d * u[j][i - 1] + (1.0 + 2.0 * d) * u[j][i] - f * u[j][i + 1] - a * q[i][j - 1]) / (a * p[i][j - 1] + b);
      });
      v[n - 1][i] = 1.0;
      RAJA::forall<Pol_Id_2_Size_1_Parent_0>(RAJA::RangeSegment{1, n - 1}, [=] (int jj) {
        int j = n - (jj + 1);
        v[j][i] = p[i][j] * v[j + 1][i] + q[i][j];
      });
    });
    RAJA::forall<Pol_Id_3_Size_1_Parent_null>(RAJA::RangeSegment{1, n - 1}, [=] (int i) {
      u[i][0] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = u[i][0];
      RAJA::forall<Pol_Id_4_Size_1_Parent_3>(RAJA::RangeSegment{1, n - 1}, [=] (int j) {
        p[i][j] = -f / (d * p[i][j - 1] + e);
        q[i][j] = (-a * v[i - 1][j] + (1.0 + 2.0 * a) * v[i][j] - c * v[i + 1][j] - d * q[i][j - 1]) / (d * p[i][j - 1] + e);
      });
      u[i][n - 1] = 1.0;
      RAJA::forall<Pol_Id_5_Size_1_Parent_3>(RAJA::RangeSegment{1, n - 1}, [=] (int jj) {
        int j = n - (jj + 1);
        u[i][j] = p[i][j] * u[i][j + 1] + q[i][j];
      });
    });
  }
}

int main() {
  int n = 2000;
  int tsteps = 1000;
  double(*u)[2000][2000];
  double(*v)[2000][2000];
  double(*p)[2000][2000];
  double(*q)[2000][2000];

  posix_memalign((void**)&u, 64, (2000) * (2000) * sizeof(double));
  posix_memalign((void**)&v, 64, (2000) * (2000) * sizeof(double));
  posix_memalign((void**)&p, 64, (2000) * (2000) * sizeof(double));
  posix_memalign((void**)&q, 64, (2000) * (2000) * sizeof(double));

  init_array(n, *u);

  DefaultTimer t;
  t.start();
  kernel_adi(tsteps, n, *u, *v, *p, *q);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("u", &(*u)[0][0], n * n);

  free((void *)u);
  free((void *)v);
  free((void *)p);
  free((void *)q);
  return 0;
}
