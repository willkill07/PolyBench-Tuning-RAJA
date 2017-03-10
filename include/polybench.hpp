#ifndef POLYBENCH_H_
#define POLYBENCH_H_

#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <cmath>

#include <string>

#include <cxxabi.h>
#include <memory>
#include <chrono>

#include <RAJA/RAJA.hxx>

#ifndef DEFAULT_DATA_READ_PATH
#define DEFAULT_DATA_PATH "/home/wkillian/defaults/"
#endif

#ifndef AUTOTUNING

using Pol_Id_0_Size_1_Parent_null = RAJA::seq_exec;
using Pol_Id_1_Size_1_Parent_null = RAJA::seq_exec;
using Pol_Id_2_Size_1_Parent_null = RAJA::seq_exec;
using Pol_Id_3_Size_1_Parent_null = RAJA::seq_exec;
using Pol_Id_5_Size_1_Parent_null = RAJA::seq_exec;
using Pol_Id_7_Size_1_Parent_null = RAJA::seq_exec;
using Pol_Id_1_Size_1_Parent_0 = RAJA::seq_exec;
using Pol_Id_2_Size_1_Parent_0 = RAJA::seq_exec;
using Pol_Id_3_Size_1_Parent_0 = RAJA::seq_exec;
using Pol_Id_2_Size_1_Parent_1 = RAJA::seq_exec;
using Pol_Id_3_Size_1_Parent_1 = RAJA::seq_exec;
using Pol_Id_3_Size_1_Parent_2 = RAJA::seq_exec;
using Pol_Id_4_Size_1_Parent_3 = RAJA::seq_exec;
using Pol_Id_5_Size_1_Parent_3 = RAJA::seq_exec;
using Pol_Id_5_Size_1_Parent_4 = RAJA::seq_exec;
using Pol_Id_6_Size_1_Parent_5 = RAJA::seq_exec;
using Pol_Id_7_Size_1_Parent_6 = RAJA::seq_exec;
using Pol_Id_8_Size_1_Parent_7 = RAJA::seq_exec;

using Pol_Id_0_Size_2_Parent_null = RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>;
using Pol_Id_1_Size_2_Parent_null = RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>;
using Pol_Id_2_Size_2_Parent_null = RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>;
using Pol_Id_3_Size_2_Parent_null = RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>;
using Pol_Id_4_Size_2_Parent_null = RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>;
using Pol_Id_9_Size_2_Parent_null = RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec>>;

using Pol_Id_0_Size_3_Parent_null = RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec,RAJA::seq_exec>>;
using Pol_Id_1_Size_3_Parent_null = RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec,RAJA::seq_exec>>;

#else

#include "config.hpp"

#endif

template <typename T>
struct Reduce {
  using type = RAJA::seq_reduce;
};

template <>
struct Reduce<RAJA::omp_parallel_for_exec> {
  using type = RAJA::omp_reduce;
};

template <typename Clock = std::chrono::steady_clock>
class Timer {
  std::chrono::time_point<Clock> begin, end;
public:
  void start() {
    begin = Clock::now();
  }
  void stop() {
    end = Clock::now();
  }
  double timeInSeconds() const {
    return std::chrono::duration<double, std::ratio<1>>(end - begin).count();
  }
};

using DefaultTimer = Timer<>;

void dumpTime(double time) {
  char filename[256];
  snprintf(filename, 256, "%s.txt", __BASE_FILE__);
  FILE* fp = fopen(filename, "w");
  fprintf(fp, "%03.12lf\n", time);
  fclose(fp);
  fp = NULL;
}

template <typename T>
static __attribute__((noinline)) void dump (const char* tag, T* data, size_t elems) {
  char filename[256];
  char* demangledType = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
  snprintf(filename, 256, "%s-%s-%s-%zu.bin", __BASE_FILE__, demangledType, tag, elems);
  free(demangledType);
  demangledType = NULL;
  FILE* fp = fopen(filename, "wb");
  fwrite(data, sizeof(T), elems, fp);
  fclose(fp);
  fp = NULL;
}

template <typename T>
static __attribute__((noinline)) void dump_init (const char* tag, T* data, size_t elems) {
  char filename[256];
  char* demangledType = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
  snprintf(filename, 256, DEFAULT_DATA_PATH "%s-%s-%s-%zu.bin", __BASE_FILE__, demangledType, tag, elems);
  free(demangledType);
  demangledType = NULL;
  FILE* fp = fopen(filename, "wb");
  if (!fp) {
    std::cerr << filename << std::endl;
  }
  fwrite(data, sizeof(T) * elems, 1, fp);
  fclose(fp);
  fp = NULL;
}

template <typename T>
static __attribute__((noinline)) bool load_init (const char* tag, T* data, size_t elems) {
  char filename[256];
  char* demangledType = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
  snprintf(filename, 256, DEFAULT_DATA_PATH "%s-%s-%s-%zu.bin", __BASE_FILE__, demangledType, tag, elems);
  free(demangledType);
  demangledType = NULL;
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    return false;
  }
  fread(data, sizeof(T) * elems, 1, fp);
  fclose(fp);
  fp = NULL;
  return true;
}

#endif
