#ifndef POLYBENCH_H_
#define POLYBENCH_H_

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include <string>

#include <chrono>
#include <cxxabi.h>
#include <memory>

#include <RAJA/RAJA.hpp>

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

using Pol_Id_0_Size_2_Parent_null =
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>;
using Pol_Id_1_Size_2_Parent_null =
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>;
using Pol_Id_2_Size_2_Parent_null =
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>;
using Pol_Id_3_Size_2_Parent_null =
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>;
using Pol_Id_4_Size_2_Parent_null =
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>;
using Pol_Id_9_Size_2_Parent_null =
    RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec>>;

using Pol_Id_0_Size_3_Parent_null = RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>>;
using Pol_Id_1_Size_3_Parent_null = RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>>;

#else

#include "config.hpp"

#endif

template <typename T> struct Reduce { using type = RAJA::seq_reduce; };

template <> struct Reduce<RAJA::omp_parallel_for_exec> {
  using type = RAJA::omp_reduce;
};

template <typename Clock = std::chrono::steady_clock> class Timer {
  std::chrono::time_point<Clock> begin, end;

public:
  void start() { begin = Clock::now(); }
  void stop() { end = Clock::now(); }
  double timeInSeconds() const {
    return std::chrono::duration<double, std::ratio<1>>(end - begin).count();
  }
};

using DefaultTimer = Timer<>;

auto getenv_or_cwd = [](char const * const ENV) -> char const *const {
  auto out = std::getenv(ENV);
  if (out)
    return out;
  return ".";
};

auto WRITE_PATH = getenv_or_cwd("OUTPUT_DIR");
auto DATA_PATH = getenv_or_cwd("DATA_DIR");

namespace detail {
  struct FileCloser {
    void operator()(FILE *ptr) { ::fclose(ptr); }
  };

  struct FreeDeleter {
    template <typename T>
    void operator()(T *ptr) { ::free(ptr); }
  };
}

std::unique_ptr<FILE, detail::FileCloser> fileHandle(char const *const filename,
                                                     char const *const mode) {
  static detail::FileCloser closer;
  return {::fopen(filename, mode), closer};
}

template <typename T> std::unique_ptr<char, detail::FreeDeleter> demangledName() {
  static detail::FreeDeleter deleter;
  return {abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr),
          deleter};
}

char const * const filenameNoDir() {
  char const * start = __BASE_FILE__;
  char const * lastSlash = start - 1;
  for (char const * i = start; *i != '\0'; ++i) {
    if (*i == '/')
      lastSlash = i;
  }
  return lastSlash + 1;
}

void dumpTime(double time) {
  char filename[256];
  snprintf(filename, 256, "%s/%s.txt", WRITE_PATH, filenameNoDir());
  auto fp = fileHandle(filename, "w");
  printf("Writing to %s\n", filename);
  fprintf(fp.get(), "%lg\n", time);
}

template <typename T>
static __attribute__((noinline)) void dump(const char *tag, T *data,
                                           size_t elems) {
  char filename[256];
  auto demangledType = demangledName<T>();
  snprintf(filename, 256, "%s/%s-%s-%s-%zu.bin", WRITE_PATH, filenameNoDir(),
           demangledType.get(), tag, elems);
  printf("Writing to %s\n", filename);
  auto fp = fileHandle(filename, "wb");
  fwrite(data, sizeof(T), elems, fp.get());
}

template <typename T>
static __attribute__((noinline)) void dump_init(const char *tag, T *data,
                                                size_t elems) {
  char filename[256];
  auto demangledType = demangledName<T>();
  snprintf(filename, 256, "%s/%s-%s-%s-%zu.bin", DATA_PATH, filenameNoDir(),
           demangledType.get(), tag, elems);
  printf("Writing to %s\n", filename);
  auto fp = fileHandle(filename, "wb");
  fwrite(data, sizeof(T) * elems, 1, fp.get());
}

template <typename T>
static __attribute__((noinline)) bool load_init(const char *tag, T *data,
                                                size_t elems) {
  char filename[256];
  auto demangledType = demangledName<T>();
  snprintf(filename, 256, "%s/%s-%s-%s-%zu.bin", DATA_PATH, filenameNoDir(),
           demangledType.get(), tag, elems);
  printf("Reading from %s\n", filename);
  auto fp = fileHandle(filename, "rb");
  if (!fp.get())
    return false;
  fread(data, sizeof(T) * elems, 1, fp.get());
  return true;
}

#endif
