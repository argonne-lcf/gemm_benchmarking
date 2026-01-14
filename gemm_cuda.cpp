#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <cmath>

// MPI Header
#include <mpi.h>

// CUDA & cuBLAS Headers
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Host BLAS Header (OpenBLAS, MKL, or Netlib)
// Ensure you link with -lblas, -lopenblas, or -lmkl_rt
#include <cblas.h>

#define TF32_MAX 3.401162134214653e+38
#define TF32_MIN 1.175494350822288e-38
#define TF32_EPSILON 9.765625000000000e-04

#ifndef ITER_MAX
#define ITER_MAX 100
#endif

#ifndef ITER_MIN
#define ITER_MIN 20
#endif

inline __half operator/(const __half& lhs, int rhs) {
    return __float2half(__half2float(lhs) / static_cast<float>(rhs));
}

inline __half operator*(const __half& lhs, double rhs) {
   return __double2half(__half2float(lhs) * rhs);
}

namespace std {
	inline __half sqrt(__half h) {
		         return __float2half(std::sqrt(__half2float(h)));
	}

	inline __half abs(const __half& h) {
			return __float2half(std::abs(__half2float(h)));
	}
}

// Communicator for a Pair of rank
// Useful for measuring Bi-Socket BW, and 2 Tile-GPU
MPI_Comm MPI_SUB_COMM;
MPI_Comm MPI_SUB_COMM_GATHER;

// CUDA Error Checking
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA API Error at line " << __LINE__ << ": "             \
                  << cudaGetErrorString(status) << std::endl;                  \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "cuBLAS API Error at line " << __LINE__ << std::endl;     \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

/*
 * CPU GEMM Wrappers (using cblas)
 */
template <typename fp_ab, typename fp_c, typename fp_scalar>
void cpu_gemm(int m, int n, int k, fp_scalar alpha, fp_ab *A, int ldA, fp_ab *B, int ldB,
              fp_scalar beta, fp_c *C_cpu, int ldC);

template <>
void cpu_gemm<double, double, double>(int m, int n, int k, double alpha, double *A, int ldA,
                                      double *B, int ldB, double beta, double *C_cpu, int ldC) {
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, ldA, B, ldB, beta,
              C_cpu, ldC);
}

template <>
void cpu_gemm<float, float, float>(int m, int n, int k, float alpha, float *A, int ldA, float *B,
                                   int ldB, float beta, float *C_cpu, int ldC) {
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, ldA, B, ldB, beta,
              C_cpu, ldC);
}

// Emulate FP16 on CPU by upcasting to FP32
template <>
void cpu_gemm<__half, __half, __half>(int m, int n, int k, __half alpha, __half *A, int ldA, __half *B,
                                   int ldB, __half beta, __half *C_cpu, int ldC) {

    std::vector<float> fA(m * k), fB(k * n), fC(m * n);
    for(int i=0; i<m*k; i++) fA[i] = __half2float(A[i]);
    for(int i=0; i<k*n; i++) fB[i] = __half2float(B[i]);
    
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, __half2float(alpha), fA.data(), ldA, 
                fB.data(), ldB, __half2float(beta), fC.data(), ldC);

    for(int i=0; i<m*n; i++) C_cpu[i] = __float2half(fC[i]);
}

#include <algorithm>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// Specialization so it doesn't take for ever
//
template <>
void cpu_gemm<int8_t, int32_t, int32_t>(int m, int n, int k,
                                        int32_t alpha, int8_t* A, int ldA,
                                        int8_t* B, int ldB,
                                        int32_t beta, int32_t* C, int ldC) {
    // C = alpha * A^T * B + beta * C
    // A^T[i,l] = A[l + i*ldA]

    constexpr int MC = 128;
    constexpr int NC = 256;
    constexpr int KC = 256;

    if (alpha == 0) {
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                C[i + j * ldC] *= beta;
        return;
    }

    #pragma omp parallel
    {
        // Thread-local buffer for packed A panel
        alignas(64) int8_t A_pack[MC * KC];

        #pragma omp for schedule(dynamic, 1)
        for (int ic = 0; ic < m; ic += MC) {
            const int mc = std::min(MC, m - ic);

            for (int pc = 0; pc < k; pc += KC) {
                const int kc = std::min(KC, k - pc);

                // Pack A^T[ic:ic+mc, pc:pc+kc] for contiguous access over i
                // Layout: A_pack[l * mc + i] = A^T[ic+i, pc+l]
                for (int l = 0; l < kc; ++l) {
                    int8_t* __restrict dst = &A_pack[l * mc];
                    for (int i = 0; i < mc; ++i) {
                        dst[i] = A[(pc + l) + (ic + i) * ldA];
                    }
                }

                for (int jc = 0; jc < n; jc += NC) {
                    const int nc = std::min(NC, n - jc);

                    for (int j = 0; j < nc; ++j) {
                        int32_t* __restrict c_ptr = &C[ic + (jc + j) * ldC];
                        const int8_t* __restrict b_ptr = &B[pc + (jc + j) * ldB];

                        // Apply beta on first K-block only
                        if (pc == 0) {
                            for (int i = 0; i < mc; ++i)
                                c_ptr[i] *= beta;
                        }

#if defined(__AVX2__)
                        // Unroll K by 4, process 8 elements at a time
                        int l = 0;
                        for (; l + 4 <= kc; l += 4) {
                            __m256i b0 = _mm256_set1_epi32(alpha * (int32_t)b_ptr[l + 0]);
                            __m256i b1 = _mm256_set1_epi32(alpha * (int32_t)b_ptr[l + 1]);
                            __m256i b2 = _mm256_set1_epi32(alpha * (int32_t)b_ptr[l + 2]);
                            __m256i b3 = _mm256_set1_epi32(alpha * (int32_t)b_ptr[l + 3]);

                            const int8_t* a0 = &A_pack[(l + 0) * mc];
                            const int8_t* a1 = &A_pack[(l + 1) * mc];
                            const int8_t* a2 = &A_pack[(l + 2) * mc];
                            const int8_t* a3 = &A_pack[(l + 3) * mc];

                            int i = 0;
                            for (; i + 8 <= mc; i += 8) {
                                // Load 8 int8 values, sign-extend to int32
                                __m256i a32_0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)(a0 + i)));
                                __m256i a32_1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)(a1 + i)));
                                __m256i a32_2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)(a2 + i)));
                                __m256i a32_3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)(a3 + i)));

                                __m256i c_vec = _mm256_loadu_si256((__m256i*)(c_ptr + i));

                                c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a32_0, b0));
                                c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a32_1, b1));
                                c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a32_2, b2));
                                c_vec = _mm256_add_epi32(c_vec, _mm256_mullo_epi32(a32_3, b3));

                                _mm256_storeu_si256((__m256i*)(c_ptr + i), c_vec);
                            }
                            // Scalar remainder
                            for (; i < mc; ++i) {
                                c_ptr[i] += (int32_t)a0[i] * (alpha * (int32_t)b_ptr[l+0])
                                          + (int32_t)a1[i] * (alpha * (int32_t)b_ptr[l+1])
                                          + (int32_t)a2[i] * (alpha * (int32_t)b_ptr[l+2])
                                          + (int32_t)a3[i] * (alpha * (int32_t)b_ptr[l+3]);
                            }
                        }
                        // K remainder
                        for (; l < kc; ++l) {
                            int32_t bval = alpha * (int32_t)b_ptr[l];
                            const int8_t* a_ptr = &A_pack[l * mc];
                            for (int i = 0; i < mc; ++i) {
                                c_ptr[i] += (int32_t)a_ptr[i] * bval;
                            }
                        }
#else
                        // Portable version with compiler auto-vectorization
                        int l = 0;
                        for (; l + 4 <= kc; l += 4) {
                            int32_t b0 = alpha * static_cast<int32_t>(b_ptr[l + 0]);
                            int32_t b1 = alpha * static_cast<int32_t>(b_ptr[l + 1]);
                            int32_t b2 = alpha * static_cast<int32_t>(b_ptr[l + 2]);
                            int32_t b3 = alpha * static_cast<int32_t>(b_ptr[l + 3]);

                            const int8_t* a0 = &A_pack[(l + 0) * mc];
                            const int8_t* a1 = &A_pack[(l + 1) * mc];
                            const int8_t* a2 = &A_pack[(l + 2) * mc];
                            const int8_t* a3 = &A_pack[(l + 3) * mc];

                            #pragma omp simd
                            for (int i = 0; i < mc; ++i) {
                                c_ptr[i] += static_cast<int32_t>(a0[i]) * b0
                                          + static_cast<int32_t>(a1[i]) * b1
                                          + static_cast<int32_t>(a2[i]) * b2
                                          + static_cast<int32_t>(a3[i]) * b3;
                            }
                        }
                        for (; l < kc; ++l) {
                            int32_t bval = alpha * static_cast<int32_t>(b_ptr[l]);
                            const int8_t* a_ptr = &A_pack[l * mc];
                            #pragma omp simd
                            for (int i = 0; i < mc; ++i) {
                                c_ptr[i] += static_cast<int32_t>(a_ptr[i]) * bval;
                            }
                        }
#endif
                    }
                }
            }
        }
    }
}


template <typename fp_ab, typename fp_c, typename fp_scalar>
void gpu_gemm(cublasHandle_t handle, int m, int n, int k, 
              const fp_scalar* alpha, const fp_ab* A, int ldA, 
              const fp_ab* B, int ldB, 
              const fp_scalar* beta, fp_c* C, int ldC);

// Double Precision Specialization
template <>
void gpu_gemm<double, double, double>(cublasHandle_t handle, int m, int n, int k, 
                                      const double* alpha, const double* A, int ldA, 
                                      const double* B, int ldB, 
                                      const double* beta, double* C, int ldC) {
    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             m, n, k, 
                             alpha, A, ldA, 
                             B, ldB, 
                             beta, C, ldC));
}

// Single Precision Specialization
template <>
void gpu_gemm<float, float, float>(cublasHandle_t handle, int m, int n, int k, 
                                   const float* alpha, const float* A, int ldA, 
                                   const float* B, int ldB, 
                                   const float* beta, float* C, int ldC) {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                             m, n, k, 
                             alpha, A, ldA, 
                             B, ldB, 
                             beta, C, ldC));
}


template <>
void gpu_gemm<__half, __half, __half>(cublasHandle_t handle, int m, int n, int k,
                                   const __half* alpha, const __half* A, int ldA,
                                   const __half* B, int ldB,
                                   const __half* beta, __half* C, int ldC) {
    CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC));
}

template <>
void gpu_gemm<int8_t, int32_t, int32_t>(cublasHandle_t handle, int m, int n, int k,
                      const int32_t* alpha, const int8_t* A, int ldA,
                      const int8_t* B, int ldB,
                      const int32_t* beta, int32_t* C, int ldC) {

    // https://docs.nvidia.com/cuda/cublas/#cublasltmatmul-regular-imma-conditions
    
    // Size
    assert (ldA % 4 == 0);
    assert (ldB % 4 == 0);
    assert (ldC % 4 == 0);
    // Alignement
    assert ( (reinterpret_cast<uintptr_t>(A) % 16) == 0);
    assert ( (reinterpret_cast<uintptr_t>(B) % 16) == 0);
    assert ( (reinterpret_cast<uintptr_t>(C) % 16) == 0);
    // T N
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              m, n, k,
                              alpha,
                              A, CUDA_R_8I, ldA,
                              B, CUDA_R_8I, ldB,
                              beta,
                              C, CUDA_R_32I, ldC,
                              CUBLAS_COMPUTE_32I, // Compute in Int32
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
  if (name == "SGEMM-TF32")
     return std::abs(x - y) <= TF32_EPSILON * std::abs(x + y) * ulp || std::abs(x - y) < TF32_MIN;
  else
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

template <> int verifyResult(__half *c_cpu, __half *c_gpu, int size, std::string name) {

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int len;
  char node_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(node_name, &len);

  int error = 0;
  for (size_t i = 0; i < size; i++) {
    float cpu = __half2float(c_cpu[i]);
    float gpu = __half2float(c_gpu[i]);


    if (!almost_equal(cpu, gpu, 100, name)) {

      if (error < 10) {
        std::cerr << "hostname " << node_name << " world_rank " << world_rank << " error number "
                  << error << " | Wrong Value. At " << i << ": " << " cpu " << cpu << " "
                  << " gpu " << gpu << " type " << name << std::endl;
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
int run(cublasHandle_t handle, int m, int n, int k, std::string name, std::string bench_type) {

  const fp_scalar alpha = fp_scalar(1.0);
  const fp_scalar beta = fp_scalar(0.0);

 
  int ldA = m;
  // Igemm assum transposed A!
  if (name == "IGEMM")
     ldA = k;

  const int ldB = k;
  const int ldC = m;

  // Allocate Device Memory
  fp_ab *A_device, *B_device;
  fp_c *C_device;
  CHECK_CUDA(cudaMalloc((void**)&A_device, m * k * sizeof(fp_ab)));
  CHECK_CUDA(cudaMalloc((void**)&B_device, k * n * sizeof(fp_ab)));
  CHECK_CUDA(cudaMalloc((void**)&C_device, m * n * sizeof(fp_c)));

  // Allocate Host Memory
  auto A_host = (fp_ab *)malloc(m * k * sizeof(fp_ab));
  auto B_host = (fp_ab *)malloc(k * n * sizeof(fp_ab));
  auto C_gpu_result = (fp_c *)malloc(m * n * sizeof(fp_c));
  auto C_cpu = (fp_c *)malloc(m * n * sizeof(fp_c));

  const fp_ab max_ab = std::numeric_limits<fp_ab>::max();

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

  // Copy Host to Device
  CHECK_CUDA(cudaMemcpy(A_device, A_host, m * k * sizeof(fp_ab), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_device, B_host, k * n * sizeof(fp_ab), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaDeviceSynchronize());

  unsigned long min_time_cpu = std::numeric_limits<unsigned long>::max();
  unsigned long min_time_gpu = std::numeric_limits<unsigned long>::max();
  int current_iter_cpu = 0;
  int current_iter_gpu = 0;

  int errors = 0;

#if defined(ENABLE_VERIFICATION)
  const int iter_to_verify_born = ITER_MAX;
#elif defined(DISABLE_VERIFICATION)
  const int iter_to_verify_born = -1;
#else // Verify only first iteration
  const int iter_to_verify_born = 1;
#endif

  for (int iter = 0, current_iter = 0; iter < ITER_MAX && current_iter < ITER_MIN; iter++) {

    if (bench_type == "cpu" || ( iter == 0 && iter < iter_to_verify_born ) ) {
      bench(&current_iter_cpu, &min_time_cpu, [&]() {
        cpu_gemm<fp_ab, fp_c, fp_scalar>(m, n, k, alpha, A_host, ldA, B_host, ldB, beta, C_cpu,
                                         ldC);
      });
    }

    if (bench_type == "gpu" || ( iter == 0  && iter < iter_to_verify_born )) {
      bench(&current_iter_gpu, &min_time_gpu, [&]() {
        gpu_gemm<fp_ab, fp_c, fp_scalar>(handle, m, n, k, &alpha, A_device, ldA,
                                              B_device, ldB, &beta, C_device, ldC);
        CHECK_CUDA(cudaDeviceSynchronize());
      });
      if (iter < iter_to_verify_born)
        CHECK_CUDA(cudaMemcpy(C_gpu_result, C_device, m * n * sizeof(fp_c), cudaMemcpyDeviceToHost));
    }

    if (bench_type == "cpu")
      current_iter = current_iter_cpu;
    else if (bench_type == "gpu")
      current_iter = current_iter_gpu;

    if (iter < iter_to_verify_born)
   	errors += verifyResult(C_cpu, C_gpu_result, m * n, name);
  }

  CHECK_CUDA(cudaFree(A_device));
  CHECK_CUDA(cudaFree(B_device));
  CHECK_CUDA(cudaFree(C_device));
  free(A_host);
  free(B_host);
  free(C_gpu_result);
  free(C_cpu);

  unsigned long min_time;
  if (bench_type == "cpu")
    min_time = min_time_cpu;
  else if (bench_type == "gpu")
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
//    std::cout << "-Memory usage " << (m*n*sizeof(fp_c)+k*n*sizeof(fp_ab)+m*k*sizeof(fp_ab)) / 1e9 << " GB" << std::endl;

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

  MPI_Init(&argc, &argv);

  int my_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Simple case, 1 mpi-rank == 1 device reported
  MPI_Comm_split(MPI_COMM_WORLD, my_rank, 0, &MPI_SUB_COMM);
  MPI_SUB_COMM_GATHER = MPI_COMM_WORLD;

  // Select Device based on Rank (simple round robin)
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  if (num_devices > 0) {
      CHECK_CUDA(cudaSetDevice(my_rank % num_devices));
  }

  // Create cuBLAS handle
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  std::string bench_type{argv[1]};

  int errors = 0;
  
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  errors += run<double, double, double>(handle, 12000, 12000, 12000, "DGEMM", bench_type);
  
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  errors += run<float, float, float>(handle, 7168 * 2, 7168 * 2, 7168 * 2, "SGEMM-FP32", bench_type);

  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
  errors += run<float, float, float>(handle, 7168 * 2, 7168 * 2, 7168 * 2, "SGEMM-TF32", bench_type);
 
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)); 
  errors += run<__half, __half, __half>(handle, 8192 * 3, 7168 * 3, 8192 * 2, "HGEMM-FP16", bench_type);

  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  errors += run<int8_t, int32_t, int32_t>(handle, 13824 * 2, 13824 * 2, 13824 * 2 , "IGEMM", bench_type);

  CHECK_CUBLAS(cublasDestroy(handle));
  MPI_Finalize();
  return errors;
}
