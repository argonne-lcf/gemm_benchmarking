#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <sycl/sycl.hpp>
#define MKL_F16 ::sycl::half
#include <mpi.h>
#include <oneapi/mkl.hpp>

#define TF32_MAX 3.401162134214653e+38
#define TF32_MIN 1.175494350822288e-38
#define TF32_EPSILON 9.765625000000000e-04
#define BFLOAT16_MAX 3.389531389251535e+38
#define BFLOAT16_MIN 1.175494350822288e-38
#define BFLOAT16_EPSILON 7.812500000000000e-03

#ifndef ITER_MAX
#define ITER_MAX 100
#endif

#ifndef ITER_MIN
#define ITER_MIN 20
#endif

// Communicator for a Pair of rank
// Useful for measuring Bi-Socket BW, and 2 Tile-GPU
MPI_Comm MPI_SUB_COMM;
MPI_Comm MPI_SUB_COMM_GATHER;

/*
 * MKL C++ Interface
 */

template <typename fp_ab, typename fp_c, typename fp_scalar>
void mkl_gemm(int m, int n, int k, fp_scalar alpha, fp_ab *A, int ldA, fp_ab *B, int ldB,
              fp_scalar beta, fp_c *C_cpu, int ldC);

template <> struct std::numeric_limits<oneapi::mkl::bfloat16> {
  static oneapi::mkl::bfloat16 max() { return BFLOAT16_MAX; }
  static oneapi::mkl::bfloat16 min() { return BFLOAT16_MIN; }
  static oneapi::mkl::bfloat16 epsilon() { return BFLOAT16_EPSILON; }
};

template <>
void mkl_gemm<MKL_F16, MKL_F16, MKL_F16>(int m, int n, int k, MKL_F16 alpha, MKL_F16 *A, int ldA,
                                         MKL_F16 *B, int ldB, MKL_F16 beta, MKL_F16 *C_cpu,
                                         int ldC) {
  cblas_hgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, ldA, B, ldB, beta,
              C_cpu, ldC);
}

template <>
void mkl_gemm<double, double, double>(int m, int n, int k, double alpha, double *A, int ldA,
                                      double *B, int ldB, double beta, double *C_cpu, int ldC) {
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, ldA, B, ldB, beta,
              C_cpu, ldC);
}

template <>
void mkl_gemm<float, float, float>(int m, int n, int k, float alpha, float *A, int ldA, float *B,
                                   int ldB, float beta, float *C_cpu, int ldC) {
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, ldA, B, ldB, beta,
              C_cpu, ldC);
}

template <>
void mkl_gemm<oneapi::mkl::bfloat16, float, float>(int m, int n, int k, float alpha,
                                                   oneapi::mkl::bfloat16 *A, int ldA,
                                                   oneapi::mkl::bfloat16 *B, int ldB, float beta,
                                                   float *C_cpu, int ldC) {
  cblas_gemm_bf16bf16f32(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, (MKL_BF16 *)A,
                         ldA, (MKL_BF16 *)B, ldB, beta, C_cpu, ldC);
}

template <>
void mkl_gemm<std::int8_t, std::int32_t, float>(int m, int n, int k, float alpha, std::int8_t *A,
                                                int ldA, std::int8_t *B, int ldB, float beta,
                                                std::int32_t *C_cpu, int ldC) {
  MKL_INT32 co = 0;
  cblas_gemm_s8u8s32(CblasColMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset, m, n, k, alpha,
                     (MKL_INT8 *)A, ldA, 0, (MKL_INT8 *)B, ldB, 0, beta, (MKL_INT32 *)C_cpu, ldC,
                     &co);
}

/*
 * Benchmark Utilities
 */

void bench(int *current_iter, unsigned long *min_time, const std::function<void()> &f) {

  MPI_Barrier(MPI_SUB_COMM);

  // Save start and end
  const unsigned long l_start = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::high_resolution_clock::now().time_since_epoch())
                                    .count();
  f();
  const unsigned long l_end = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::high_resolution_clock::now().time_since_epoch())
                                  .count();

  unsigned long start, end;
  MPI_Allreduce(&l_start, &start, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_SUB_COMM);
  MPI_Allreduce(&l_end, &end, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_SUB_COMM);

  unsigned long time = end - start;

  if (time >= *min_time) {
    *current_iter = *current_iter + 1;
  } else {
    *current_iter = 0;
    *min_time = time;
  }
}

template <typename fp> bool almost_equal(fp x, fp y, int ulp, std::string name) {
  // TF32 is not a type
  if (name == "SGEMM-TF32") {
    return std::abs(x - y) <= TF32_EPSILON * std::abs(x + y) * ulp || std::abs(x - y) < TF32_MIN;
  } else
    return std::abs(x - y) <= std::numeric_limits<fp>::epsilon() * std::abs(x + y) * ulp ||
           std::abs(x - y) < std::numeric_limits<fp>::min();
}

template <typename fp> int verifyResult(fp *c_cpu, fp *c_gpu, int size, std::string name) {

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int len;
  char node_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(node_name, &len);

  int error = 0;
  for (size_t i = 0; i < size; i++) {
    if (!almost_equal(c_cpu[i], c_gpu[i], 100, name)) {

      if (error < 10) {
        std::cerr << "hostname " << node_name << " world_rank " << world_rank << " error number "
                  << error << " | Wrong Value. At " << i << ": " << " cpu " << (fp)c_cpu[i] << " "
                  << " gpu " << (fp)c_gpu[i] << " type " << name << std::endl;
      }

      error++;
    }
  }
  if (error > 0)
    std::cerr << "Failed!" << std::endl;

  return error;
}

template <typename T1, typename T2> typename T1::value_type quant(const T1 &x, T2 q) {
  assert(q >= 0.0 && q <= 1.0);

  const auto n = x.size();
  const auto id = (n - 1) * q;
  const auto lo = floor(id);
  const auto hi = ceil(id);
  const auto qs = x[lo];
  const auto h = (id - lo);

  return (1.0 - h) * qs + h * x[hi];
}

template <typename fp_ab, typename fp_c, typename fp_scalar>
int run(sycl::queue Q, int m, int n, int k, std::string name, std::string bench_type,
        oneapi::mkl::blas::compute_mode mode = oneapi::mkl::blas::compute_mode::standard) {

  auto transA = oneapi::mkl::transpose::nontrans;
  auto transB = oneapi::mkl::transpose::nontrans;

  fp_scalar alpha = fp_scalar(1.0);
  fp_scalar beta = fp_scalar(0.0);
  int ldA = m;
  int ldB = k;
  int ldC = m;

  auto A_device = sycl::malloc_device<fp_ab>(m * k, Q);
  auto B_device = sycl::malloc_device<fp_ab>(k * n, Q);
  auto C_device = sycl::malloc_device<fp_c>(m * n, Q);

  auto A_host = (fp_ab *)malloc(m * k * sizeof(fp_ab));
  auto B_host = (fp_ab *)malloc(k * n * sizeof(fp_ab));
  auto C_gpu_result = (fp_c *)malloc(m * n * sizeof(fp_c));

  auto C_cpu = (fp_c *)malloc(m * n * sizeof(fp_c));

  fp_ab max_ab = std::numeric_limits<fp_ab>::max();

  std::srand(0);

  fp_c max_c_array_value = std::sqrt(std::numeric_limits<fp_c>::max() / k);
  // assumes fp_c is bigger
  fp_ab max_array_value = std::min((fp_c)max_c_array_value, (fp_c)max_ab / 2);
  for (size_t i = 0; i < (m * k); i++) {
    A_host[i] = fp_ab(max_array_value) * double((std::rand() / (double)RAND_MAX));
  }
  for (size_t i = 0; i < (k * n); i++) {
    B_host[i] = fp_ab(max_array_value) * double((std::rand() / (double)RAND_MAX));
  }

  Q.copy(A_host, A_device, m * k).wait();
  Q.copy(B_host, B_device, k * n).wait();

  unsigned long min_time_cpu = std::numeric_limits<unsigned long>::max();
  unsigned long min_time_gpu = std::numeric_limits<unsigned long>::max();
  int current_iter_cpu = 0;
  int current_iter_gpu = 0;

  int errors = 0;

  for (int iter = 0, current_iter = 0; iter < ITER_MAX && current_iter < ITER_MIN; iter++) {

    if (bench_type == "cpu" || iter == 0) {
      bench(&current_iter_cpu, &min_time_cpu, [&]() {
        mkl_gemm<fp_ab, fp_c, fp_scalar>(m, n, k, alpha, A_host, ldA, B_host, ldB, beta, C_cpu,
                                         ldC);
      });
    }

    if (bench_type == "gpu" || iter == 0) {
      bench(&current_iter_gpu, &min_time_gpu, [&]() {
        oneapi::mkl::blas::column_major::gemm(Q, transA, transB, m, n, k, alpha, A_device, ldA,
                                              B_device, ldB, beta, C_device, ldC, mode)
            .wait();
      });
      Q.copy(C_device, C_gpu_result, m * n).wait();
    }

    if (bench_type == "cpu")
      current_iter = current_iter_cpu;
    if (bench_type == "gpu")
      current_iter = current_iter_gpu;

#ifndef AVOID_VERIFICATION
    errors += verifyResult(C_cpu, C_gpu_result, m * n, name);
#endif
  }

  free(A_device, Q);
  free(B_device, Q);
  free(C_device, Q);
  free(A_host);
  free(B_host);
  free(C_gpu_result);
  free(C_cpu);

  unsigned long min_time;
  if (bench_type == "cpu")
    min_time = min_time_cpu;
  if (bench_type == "gpu")
    min_time = min_time_gpu;

  // Now do a gather
  int root_rank = 0;
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == root_rank) {
    int gather_size;
    MPI_Comm_size(MPI_SUB_COMM_GATHER, &gather_size);

    std::vector<double> flops(gather_size);
    {
      std::vector<unsigned long> min_times(gather_size);
      MPI_Gather(&min_time, 1, MPI_UNSIGNED_LONG, min_times.data(), 1, MPI_UNSIGNED_LONG, root_rank,
                 MPI_SUB_COMM_GATHER);
      {
        int sub_size;
        MPI_Comm_size(MPI_SUB_COMM, &sub_size);
        std::transform(min_times.begin(), min_times.end(), flops.begin(),
                       [&](unsigned long val) { return (2. * m * n * k * sub_size) / val; });
      }
#ifdef SAVE
      {
        std::string filename = name + ".txt";
        std::ofstream fout(filename.c_str());
        for (auto const &x : flops)
          fout << x << '\n';
      }
#endif
      std::sort(flops.begin(), flops.end());
    }
    std::cout << "Result For " << name << " (sample size: " << gather_size << ")" << std::endl;
    std::cout << "-Min " << flops.front() << " GFlop/s" << std::endl;
    std::cout << "-Q1 " << quant(flops, 0.25) << " GFlop/s" << std::endl;
    std::cout << "-Q2(median) " << quant(flops, 0.50) << " GFlop/s" << std::endl;
    std::cout << "-Q3 " << quant(flops, 0.75) << " GFlop/s" << std::endl;
    std::cout << "-Max " << flops.back() << " GFlop/s" << std::endl;
    std::cout << "-Mem " << (m*n*sizeof(fp_c)+k*n*sizeof(fp_ab)+m*k*sizeof(fp_ab)) / 1e9 << " GB" << std::endl;

  } else if (MPI_SUB_COMM_GATHER != MPI_COMM_NULL) {
    MPI_Gather(&min_time, 1, MPI_UNSIGNED_LONG, NULL, 0, MPI_UNSIGNED_LONG, root_rank,
               MPI_SUB_COMM_GATHER);
  }

  int mpi_errors = 0;
  MPI_Reduce(&errors, &mpi_errors, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  return mpi_errors;
}

/*
 * Main
 */

int main(int argc, char **argv) {

  MPI_Init(NULL, NULL);

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  std::string bench_type{argv[1]};

  if (bench_type == "gpu") {
    // Best of two Tiles
    MPI_Comm_split(MPI_COMM_WORLD, my_rank / 2, 0, &MPI_SUB_COMM);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::vector<int> ranks(world_size / 2);
    {
      int n = -2;
      std::generate(ranks.begin(), ranks.end(), [&n] { return n += 2; });
    }
    {
      MPI_Group world_group;
      MPI_Comm_group(MPI_COMM_WORLD, &world_group);
      MPI_Group new_group;
      MPI_Group_incl(world_group, ranks.size(), ranks.data(), &new_group);
      MPI_Comm_create(MPI_COMM_WORLD, new_group, &MPI_SUB_COMM_GATHER);
    }
  } else if (bench_type == "cpu") {
    MPI_Comm_split(MPI_COMM_WORLD, my_rank, 0, &MPI_SUB_COMM);
    MPI_SUB_COMM_GATHER = MPI_COMM_WORLD;
  }

  sycl::queue Q;
  int errors = 0;
  // Stream said 4Time LLC per Array ~800 * 3 ~ Total Memory FootPrint = 2.4 G for GPU. for CPU: ~ 1
  // GB * 3  = 3 GB we can also  chosen size based on where the GEMM flop-rate levels offs for most
  // GEMMs (as in GEMM_sizes.csv) Or even do to a sweep at runtime. Right now we hardcode some size

  errors += run<double, double, double>(Q, 12000, 12000, 12000, "DGEMM", bench_type);

  errors += run<float, float, float>(Q, 7168 * 2, 7168 * 2, 7168 * 2, "SGEMM-FP32", bench_type);

  if (bench_type != "cpu")
    errors += run<float, float, float>(Q, 7168 * 2, 7168 * 2, 7168 * 2, "SGEMM-TF32", bench_type,
                                       oneapi::mkl::blas::compute_mode::float_to_tf32);

  errors += run<oneapi::mkl::bfloat16, float, float>(Q, 8192 * 3, 7168 * 3, 8192 * 2,
                                                     "HGEMM-BF16", bench_type);

  // Small Footprint reflecting more how application are using it
  errors +=
      run<sycl::half, sycl::half, sycl::half>(Q, 12000, 12000, 12000, "HGEMM-FP16", bench_type);

  errors +=
      run<std::int8_t, std::int32_t, float>(Q, 13824 * 2, 13824 * 2, 13824, "IGEMM", bench_type);

  MPI_Finalize();
  return errors;
}
