#include "polybench.hpp"

typedef char base;

static __attribute__((noinline)) void init_array(int n, base seq[5500], int table[5500][5500]) {
  int i, j;

  if (load_init("seq", seq, n) &&
      load_init("table", table[0], n * n))
    return;

  for (i = 0; i < n; i++) {
    seq[i] = (base)((i + 1) % 4);
  }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      table[i][j] = 0;

  dump_init("seq", seq, n);
  dump_init("table", table[0], n * n);
}

static __attribute__ ((noinline)) void kernel_nussinov(int n, base seq[5500], int table[5500][5500]) {
  RAJA::forall<Pol_Id_0_Size_1_Parent_null>(RAJA::RangeSegment{0, n}, [=] (int ii) {
    int i = (n - 1) - ii;
    RAJA::forall<Pol_Id_1_Size_1_Parent_0>(RAJA::RangeSegment{i + 1, n}, [=] (int j) {
      if (j - 1 >= 0)
        table[i][j] = ((table[i][j] >= table[i][j - 1]) ? table[i][j] : table[i][j - 1]);
      if (i + 1 < n)
        table[i][j] = ((table[i][j] >= table[i + 1][j]) ? table[i][j] : table[i + 1][j]);
      if (j - 1 >= 0 && i + 1 < n) {
        if (i < j - 1)
          table[i][j] = ((table[i][j] >= table[i + 1][j - 1] + (((seq[i]) + (seq[j])) == 3 ? 1 : 0)) ? table[i][j] : table[i + 1][j - 1] + (((seq[i]) + (seq[j])) == 3 ? 1 : 0));
        else
          table[i][j] = ((table[i][j] >= table[i + 1][j - 1]) ? table[i][j] : table[i + 1][j - 1]);
      }
      RAJA::forall<Pol_Id_2_Size_1_Parent_1>(RAJA::RangeSegment{i + 1, j}, [=] (int k) {
        table[i][j] = ((table[i][j] >= table[i][k] + table[k + 1][j]) ? table[i][j] : table[i][k] + table[k + 1][j]);
      });
    });
  });
}

int main() {
  int n = 5500;
  base(*seq)[5500];
  int(*table)[5500][5500];

  posix_memalign((void**)&seq, 64, 5500 * sizeof(base));
  posix_memalign((void**)&table, 64, (5500) * (5500) * sizeof(int));

  init_array(n, *seq, *table);

  DefaultTimer t;
  t.start();
  kernel_nussinov(n, *seq, *table);
  t.stop();
  dumpTime(t.timeInSeconds());

  dump("table", &(*table)[0][0], n * n);

  free((void *)seq);
  free((void *)table);
  return 0;
}
