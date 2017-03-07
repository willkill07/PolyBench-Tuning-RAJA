#include "polybench.hpp"

typedef char base;

static __attribute__((noinline)) void init_array(int n, base seq[5500], int table[5500][5500]) {
  int i, j;
  for (i = 0; i < n; i++) {
    seq[i] = (base)((i + 1) % 4);
  }
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      table[i][j] = 0;
}

static __attribute__ ((noinline)) void kernel_nussinov(int n, base seq[5500], int table[5500][5500]) {
  int i, j, k;
  for (i = n - 1; i >= 0; i--) {
    for (j = i + 1; j < n; j++) {
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
      for (k = i + 1; k < j; k++) {
        table[i][j] = ((table[i][j] >= table[i][k] + table[k + 1][j]) ? table[i][j] : table[i][k] + table[k + 1][j]);
      }
    }
  }
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
