// 1. 补充必要头文件（增强依赖完整性，避免隐式依赖问题）
#include <cuda_runtime_api.h>  // 更完整的 CUDA 运行时 API
#include <device_launch_parameters.h>
#include <algorithm>
#include <stdexcept>           // 标准异常类（替代部分 runtime_error）
#include <utility>             // 确保 STL 基础依赖
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <sstream>
namespace needle {
namespace cuda {

// 2. 宏定义添加注释，增强可维护性
#define BASE_THREAD_NUM 256    // CUDA 核函数默认块大小（适配多数 GPU 架构）
#define TILE 4                 // 矩阵乘法 Tile 大小（分块计算粒度）
#define MAX_VEC_SIZE 8         // 最大支持维度（避免设备端数组越界）

using scalar_t = float;        // 统一数据类型（便于后续修改为 double 等）
const size_t ELEM_SIZE = sizeof(scalar_t);

// 3. 前向声明添加核函数属性（帮助编译器优化寄存器分配）
__global__ void __launch_bounds__(BASE_THREAD_NUM) FillKernel(scalar_t* out, scalar_t val, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) CompactKernel(const scalar_t* a, scalar_t* out, size_t size, struct CudaVec shape, struct CudaVec strides, size_t offset);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, struct CudaVec shape, struct CudaVec strides, size_t offset);
__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, struct CudaVec shape, struct CudaVec strides, size_t offset);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarMulKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarDivKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarPowerKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarMaximumKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarEqKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarGeKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size);
__global__ void __launch_bounds__(TILE* TILE) MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P);  // 适配 Tile 大小
__global__ void __launch_bounds__(BASE_THREAD_NUM) ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size);
__global__ void __launch_bounds__(BASE_THREAD_NUM) ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size);

// 4. CudaArray 类：增强 const 正确性 + 显式析构说明 + 错误提示优化
struct CudaArray {
    // 构造函数：强化错误信息（指明是设备内存分配失败）
    CudaArray(const size_t size) : size(size) {
        auto deleter = [](scalar_t* ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                fprintf(stderr, "Warning: CUDA free failed: %s\n", cudaGetErrorString(err));
            }
        };
        scalar_t* raw_ptr = nullptr;
        cudaError_t err = cudaMalloc(&raw_ptr, size * ELEM_SIZE);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA device memory allocation failed: ") + cudaGetErrorString(err) +
                " (requested size: " + std::to_string(size * ELEM_SIZE) + " bytes)"
            );
        }
        ptr = std::shared_ptr<scalar_t>(raw_ptr, deleter);
    }

    // 显式声明析构（虽由 shared_ptr 管理，但增强代码可读性）
    ~CudaArray() = default;

    // 设备指针转整数：添加类型安全转换 + 注释
    size_t ptr_as_int() const {
        // 返回设备内存指针的整数表示（用于调试或外部指针传递）
        return static_cast<size_t>(reinterpret_cast<uintptr_t>(ptr.get()));
    }

    // 获取指针：区分 const/非 const 版本（增强类型安全）
    scalar_t* get_ptr() { return ptr.get(); }
    const scalar_t* get_ptr() const { return ptr.get(); }

    std::shared_ptr<scalar_t> ptr;  // 设备内存智能指针（自动释放）
    const size_t size;              // 数组元素个数（const 不可修改）
};

// 5. CudaDims 结构体：添加构造辅助函数（减少重复代码）
struct CudaDims {
    dim3 block, grid;

    // 静态方法：快速创建 1D 维度配置
    static CudaDims one_dim(size_t elem_count) {
        CudaDims dim;
        dim.block = dim3(BASE_THREAD_NUM, 1, 1);
        dim.grid = dim3((elem_count + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM, 1, 1);
        return dim;
    }

    // 静态方法：快速创建矩阵乘法 2D 维度配置（适配 Tile 大小）
    static CudaDims matmul_2d(uint32_t M, uint32_t P) {
        CudaDims dim;
        dim.block = dim3(TILE, TILE, 1);
        dim.grid = dim3((M + TILE - 1) / TILE, (P + TILE - 1) / TILE, 1);
        return dim;
    }
};

// 6. CudaVec 结构体：添加边界检查（避免维度超限）
struct CudaVec {
    uint32_t size;
    int32_t data[MAX_VEC_SIZE];

    // 辅助函数：从 std::vector 初始化 + 维度超限检查
    static CudaVec from_std(const std::vector<int32_t>& vec) {
        if (vec.size() > MAX_VEC_SIZE) {
            throw std::invalid_argument(
                "CUDA dimension limit exceeded: max supported dimensions = " + std::to_string(MAX_VEC_SIZE) +
                ", requested = " + std::to_string(vec.size())
            );
        }
        CudaVec res;
        res.size = static_cast<uint32_t>(vec.size());
        std::copy(vec.begin(), vec.end(), res.data);
        return res;
    }
};

// 7. 核函数实现：添加错误检查 + 性能微调
__global__ void __launch_bounds__(BASE_THREAD_NUM) FillKernel(scalar_t* out, scalar_t val, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[gid] = val;
    }
}

// Fill 函数：添加核函数启动错误检查
void Fill(CudaArray* out, scalar_t val) {
    if (!out) throw std::invalid_argument("Fill: out array cannot be null");
    CudaDims dim = CudaDims::one_dim(out->size);
    FillKernel<<<dim.grid, dim.block>>>(out->get_ptr(), val, out->size);
    // 检查核函数启动错误（避免静默失败）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Fill kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// 设备端索引计算：添加 inline 优化 + 注释
__device__ inline size_t gid_to_idx(size_t gid, CudaVec shape, CudaVec strides, size_t offset) {
    size_t idx = offset;
    // 从最高维到最低维计算索引（适配多维数组紧凑化）
    for (int32_t i = static_cast<int32_t>(shape.size) - 1; i >= 0; --i) {
        idx += (gid % shape.data[i]) * strides.data[i];
        gid /= shape.data[i];
    }
    return idx;
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[gid] = a[gid_to_idx(gid, shape, strides, offset)];
    }
}

// Compact 函数：参数改为 const 引用（减少拷贝）+ 错误检查
void Compact(const CudaArray& a, CudaArray* out, const std::vector<int32_t>& shape, const std::vector<int32_t>& strides, size_t offset) {
    if (!out) throw std::invalid_argument("Compact: out array cannot be null");
    if (out->size == 0) throw std::invalid_argument("Compact: out array size cannot be zero");
    CudaDims dim = CudaDims::one_dim(out->size);
    CudaVec cuda_shape = CudaVec::from_std(shape);
    CudaVec cuda_strides = CudaVec::from_std(strides);
    CompactKernel<<<dim.grid, dim.block>>>(a.get_ptr(), out->get_ptr(), out->size, cuda_shape, cuda_strides, offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Compact kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t idx = gid_to_idx(gid, shape, strides, offset);
        out[idx] = a[gid];
    }
}

// EwiseSetitem 函数：修复核函数维度（用 a.size 而非 out->size）+ 错误检查
void EwiseSetitem(const CudaArray& a, CudaArray* out, const std::vector<int32_t>& shape, const std::vector<int32_t>& strides, size_t offset) {
    if (!out) throw std::invalid_argument("EwiseSetitem: out array cannot be null");
    if (a.size == 0) throw std::invalid_argument("EwiseSetitem: a array size cannot be zero");
    CudaDims dim = CudaDims::one_dim(a.size);  // 核函数维度应匹配输入 a 的大小
    CudaVec cuda_shape = CudaVec::from_std(shape);
    CudaVec cuda_strides = CudaVec::from_std(strides);
    EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.get_ptr(), out->get_ptr(), a.size, cuda_shape, cuda_strides, offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("EwiseSetitem kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarSetitemKernel(const scalar_t val, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t idx = gid_to_idx(gid, shape, strides, offset);
        out[idx] = val;
    }
}

// ScalarSetitem 函数：添加 size 边界检查（避免越界）
void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, const std::vector<int32_t>& shape, const std::vector<int32_t>& strides, size_t offset) {
    if (!out) throw std::invalid_argument("ScalarSetitem: out array cannot be null");
    if (size == 0) throw std::invalid_argument("ScalarSetitem: size cannot be zero");
    if (size > out->size) throw std::out_of_range("ScalarSetitem: size exceeds out array size");
    CudaDims dim = CudaDims::one_dim(size);
    CudaVec cuda_shape = CudaVec::from_std(shape);
    CudaVec cuda_strides = CudaVec::from_std(strides);
    ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->get_ptr(), size, cuda_shape, cuda_strides, offset);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("ScalarSetitem kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// 8. 元素级操作函数：统一错误检查逻辑（以 EwiseAdd 为例，其他类似）
__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        out[gid] = a[gid] + b[gid];
    }
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    if (!out) throw std::invalid_argument("EwiseAdd: out array cannot be null");
    if (a.size != b.size || a.size != out->size) {
        throw std::invalid_argument("EwiseAdd: array sizes mismatch: a=" + std::to_string(a.size) +
                                    ", b=" + std::to_string(b.size) + ", out=" + std::to_string(out->size));
    }
    CudaDims dim = CudaDims::one_dim(out->size);
    EwiseAddKernel<<<dim.grid, dim.block>>>(a.get_ptr(), b.get_ptr(), out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("EwiseAdd kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// （注：其他元素级函数 EwiseMul/EwiseDiv/ScalarAdd 等均按 EwiseAdd 逻辑添加错误检查，此处省略重复代码）
__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = a[gid] + val;
}
void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
    if (!out) throw std::invalid_argument("ScalarAdd: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("ScalarAdd: a and out size mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    ScalarAddKernel<<<dim.grid, dim.block>>>(a.get_ptr(), val, out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("ScalarAdd kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = a[gid] * b[gid];
}
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    if (!out) throw std::invalid_argument("EwiseMul: out array cannot be null");
    if (a.size != b.size || a.size != out->size) throw std::invalid_argument("EwiseMul: array sizes mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    EwiseMulKernel<<<dim.grid, dim.block>>>(a.get_ptr(), b.get_ptr(), out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("EwiseMul kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarMulKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = a[gid] * val;
}
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
    if (!out) throw std::invalid_argument("ScalarMul: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("ScalarMul: a and out size mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    ScalarMulKernel<<<dim.grid, dim.block>>>(a.get_ptr(), val, out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("ScalarMul kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = a[gid] / b[gid];
}
void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    if (!out) throw std::invalid_argument("EwiseDiv: out array cannot be null");
    if (a.size != b.size || a.size != out->size) throw std::invalid_argument("EwiseDiv: array sizes mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    EwiseDivKernel<<<dim.grid, dim.block>>>(a.get_ptr(), b.get_ptr(), out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("EwiseDiv kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarDivKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = a[gid] / val;
}
void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
    if (!out) throw std::invalid_argument("ScalarDiv: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("ScalarDiv: a and out size mismatch");
    if (val == 0) throw std::domain_error("ScalarDiv: division by zero");
    CudaDims dim = CudaDims::one_dim(out->size);
    ScalarDivKernel<<<dim.grid, dim.block>>>(a.get_ptr(), val, out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("ScalarDiv kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarPowerKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = pow(a[gid], val);
}
void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
    if (!out) throw std::invalid_argument("ScalarPower: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("ScalarPower: a and out size mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    ScalarPowerKernel<<<dim.grid, dim.block>>>(a.get_ptr(), val, out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("ScalarPower kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = fmax(a[gid], b[gid]);
}
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    if (!out) throw std::invalid_argument("EwiseMaximum: out array cannot be null");
    if (a.size != b.size || a.size != out->size) throw std::invalid_argument("EwiseMaximum: array sizes mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.get_ptr(), b.get_ptr(), out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("EwiseMaximum kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarMaximumKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = fmax(a[gid], val);
}
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
    if (!out) throw std::invalid_argument("ScalarMaximum: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("ScalarMaximum: a and out size mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.get_ptr(), val, out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("ScalarMaximum kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = (a[gid] == b[gid]) ? 1.0f : 0.0f;  // 统一返回 0/1 浮点数，避免未定义值
}
void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    if (!out) throw std::invalid_argument("EwiseEq: out array cannot be null");
    if (a.size != b.size || a.size != out->size) throw std::invalid_argument("EwiseEq: array sizes mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    EwiseEqKernel<<<dim.grid, dim.block>>>(a.get_ptr(), b.get_ptr(), out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("EwiseEq kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarEqKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = (a[gid] == val) ? 1.0f : 0.0f;
}
void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
    if (!out) throw std::invalid_argument("ScalarEq: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("ScalarEq: a and out size mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    ScalarEqKernel<<<dim.grid, dim.block>>>(a.get_ptr(), val, out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("ScalarEq kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = (a[gid] >= b[gid]) ? 1.0f : 0.0f;
}
void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    if (!out) throw std::invalid_argument("EwiseGe: out array cannot be null");
    if (a.size != b.size || a.size != out->size) throw std::invalid_argument("EwiseGe: array sizes mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    EwiseGeKernel<<<dim.grid, dim.block>>>(a.get_ptr(), b.get_ptr(), out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("EwiseGe kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) ScalarGeKernel(const scalar_t* a, const scalar_t val, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = (a[gid] >= val) ? 1.0f : 0.0f;
}
void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
    if (!out) throw std::invalid_argument("ScalarGe: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("ScalarGe: a and out size mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    ScalarGeKernel<<<dim.grid, dim.block>>>(a.get_ptr(), val, out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("ScalarGe kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = (a[gid] > 0) ? log(a[gid]) : -INFINITY;  // 处理负输入（返回负无穷，避免 NaN 静默传播）
}
void EwiseLog(const CudaArray& a, CudaArray* out) {
    if (!out) throw std::invalid_argument("EwiseLog: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("EwiseLog: a and out size mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    EwiseLogKernel<<<dim.grid, dim.block>>>(a.get_ptr(), out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("EwiseLog kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = exp(a[gid]);
}
void EwiseExp(const CudaArray& a, CudaArray* out) {
    if (!out) throw std::invalid_argument("EwiseExp: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("EwiseExp: a and out size mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    EwiseExpKernel<<<dim.grid, dim.block>>>(a.get_ptr(), out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("EwiseExp kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) out[gid] = tanh(a[gid]);
}
void EwiseTanh(const CudaArray& a, CudaArray* out) {
    if (!out) throw std::invalid_argument("EwiseTanh: out array cannot be null");
    if (a.size != out->size) throw std::invalid_argument("EwiseTanh: a and out size mismatch");
    CudaDims dim = CudaDims::one_dim(out->size);
    EwiseTanhKernel<<<dim.grid, dim.block>>>(a.get_ptr(), out->get_ptr(), out->size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("EwiseTanh kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

// 9. 矩阵乘法：优化块维度创建 + 核函数注释
__global__ void __launch_bounds__(TILE* TILE) MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P) {
    // 2D 线程索引（对应输出矩阵的 (x,y) 位置）
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < P) {
        scalar_t sum = 0.0f;
        // 按 Tile 粒度计算（当前为朴素实现，可后续优化共享内存）
        for (uint32_t k = 0; k < N; ++k) {
            sum += a[x * N + k] * b[k * P + y];
        }
        out[x * P + y] = sum;
    }
}
void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N, uint32_t P) {
    if (!out) throw std::invalid_argument("Matmul: out array cannot be null");
    // 检查矩阵维度兼容性：a(MxN) * b(NxP) = out(MxP)
    if (a.size != static_cast<size_t>(M * N)) throw std::invalid_argument("Matmul: a size mismatch with M*N");
    if (b.size != static_cast<size_t>(N * P)) throw std::invalid_argument("Matmul: b size mismatch with N*P");
    if (out->size != static_cast<size_t>(M * P)) throw std::invalid_argument("Matmul: out size mismatch with M*P");
    CudaDims dim = CudaDims::matmul_2d(M, P);  // 用辅助函数创建 2D 维度
    MatmulKernel<<<dim.grid, dim.block>>>(a.get_ptr(), b.get_ptr(), out->get_ptr(), M, N, P);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("Matmul kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

// 10. 归约操作：添加输入有效性检查
__global__ void __launch_bounds__(BASE_THREAD_NUM) ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t start = gid * reduce_size;
        scalar_t max_val = a[start];
        for (size_t i = 1; i < reduce_size; ++i) {
            max_val = fmax(max_val, a[start + i]);
        }
        out[gid] = max_val;
    }
}
void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
    if (!out) throw std::invalid_argument("ReduceMax: out array cannot be null");
    if (reduce_size == 0) throw std::invalid_argument("ReduceMax: reduce_size cannot be zero");
    if (a.size != out->size * reduce_size) throw std::invalid_argument("ReduceMax: a size must be out.size * reduce_size");
    CudaDims dim = CudaDims::one_dim(out->size);
    ReduceMaxKernel<<<dim.grid, dim.block>>>(a.get_ptr(), out->get_ptr(), out->size, reduce_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("ReduceMax kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

__global__ void __launch_bounds__(BASE_THREAD_NUM) ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t start = gid * reduce_size;
        scalar_t sum = a[start];
        for (size_t i = 1; i < reduce_size; ++i) {
            sum += a[start + i];
        }
        out[gid] = sum;
    }
}
void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
    if (!out) throw std::invalid_argument("ReduceSum: out array cannot be null");
    if (reduce_size == 0) throw std::invalid_argument("ReduceSum: reduce_size cannot be zero");
    if (a.size != out->size * reduce_size) throw std::invalid_argument("ReduceSum: a size must be out.size * reduce_size");
    CudaDims dim = CudaDims::one_dim(out->size);
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.get_ptr(), out->get_ptr(), out->size, reduce_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error("ReduceSum kernel launch failed: " + std::string(cudaGetErrorString(err)));
}

} // namespace cuda
} // namespace needle

// 11. Pybind11 绑定：增强文档字符串 + 支持负索引 + 版本信息
PYBIND11_MODULE(CUDA_BACKEND, m) {
    namespace py = pybind11;
    using namespace needle;
    using namespace cuda;

    // 模块元信息（便于用户识别版本和设备）
    m.attr("__version__") = py::str("1.0.0");
    m.attr("__device__name__") = py::str("cuda");
    m.attr("__tile_size__") = py::int_(TILE);
    m.attr("__max_dimensions__") = py::int_(MAX_VEC_SIZE);

    // 绑定 std::vector<int32_t>：支持负索引 + 详细文档
    py::class_<std::vector<int32_t>>(m, "Int32Vector", "Vector of int32_t (for CUDA dimension/stride inputs)")
        .def(py::init<>(), "Create an empty Int32Vector")
        .def("push_back", (void (std::vector<int32_t>::*)(const int32_t&)) &std::vector<int32_t>::push_back,
             py::arg("value"), "Add an int32 value to the end of the vector")
        .def("size", &std::vector<int32_t>::size, "Get the number of elements in the vector")
        .def("__getitem__", [](const std::vector<int32_t>& v, int64_t i) {
            // 支持负索引（-1 表示最后一个元素）
            if (i < 0) i += static_cast<int64_t>(v.size());
            if (i < 0 || i >= static_cast<int64_t>(v.size())) {
                throw py::index_error("Int32Vector index out of range: " + std::to_string(i));
            }
            return v[static_cast<size_t>(i)];
        }, py::arg("index"), py::return_value_policy::copy, "Get element by index (supports negative indices)")
        .def("clear", &std::vector<int32_t>::clear, "Clear all elements in the vector");

    // 绑定 std::vector<size_t>：同理支持负索引
    py::class_<std::vector<size_t>>(m, "SizeTVector", "Vector of size_t (for numpy shape/stride inputs)")
        .def(py::init<>(), "Create an empty SizeTVector")
        .def("push_back", (void (std::vector<size_t>::*)(const size_t&)) &std::vector<size_t>::push_back,
             py::arg("value"), "Add a size_t value to the end of the vector")
        .def("size", &std::vector<size_t>::size, "Get the number of elements in the vector")
        .def("__getitem__", [](const std::vector<size_t>& v, int64_t i) {
            if (i < 0) i += static_cast<int64_t>(v.size());
            if (i < 0 || i >= static_cast<int64_t>(v.size())) {
                throw py::index_error("SizeTVector index out of range: " + std::to_string(i));
            }
            return v[static_cast<size_t>(i)];
        }, py::arg("index"), py::return_value_policy::copy, "Get element by index (supports negative indices)")
        .def("clear", &std::vector<size_t>::clear, "Clear all elements in the vector");

    // 绑定 CudaArray 类：详细文档说明
   py::class_<CudaArray>(m, "Array", "CUDA device array (stores float32 data)")
    .def(py::init<size_t>(), py::arg("size"), "Create a CUDA array with given element count", py::return_value_policy::take_ownership)
    .def_readonly("size", &CudaArray::size, "Number of float32 elements in the array")
    .def("ptr", &CudaArray::ptr_as_int, "Get CUDA device pointer as integer (for debugging)")
    .def("__repr__", [](const CudaArray& arr) {
        std::stringstream ss;  // 用字符串流构建输出
        ss << "<CUDA Array: size=" << arr.size
           << ", device_ptr=0x" << std::hex << arr.ptr_as_int() << std::dec << ">";
        return ss.str();  // 转换为字符串返回
    }, "String representation of the CUDA array");

    // 绑定核心函数：添加参数文档 + 错误说明
    m.def("fill", &Fill, py::arg("out_array"), py::arg("value"),
          "Fill CUDA array with a scalar float value\n"
          "Args:\n"
          "  out_array: Target CUDA Array (will be modified)\n"
          "  value: Scalar float to fill");

    m.def("compact", &Compact, py::arg("in_array"), py::arg("out_array"), py::arg("shape"), py::arg("strides"), py::arg("offset"),
          "Compact multi-dimensional array to 1D (device-side)\n"
          "Args:\n"
          "  in_array: Input CUDA Array (multi-dimensional)\n"
          "  out_array: Output CUDA Array (1D, same size as in_array)\n"
          "  shape: Int32Vector of input array dimensions\n"
          "  strides: Int32Vector of input array strides (elements per step)\n"
          "  offset: Starting element offset in input array");

    m.def("ewise_setitem", &EwiseSetitem, py::arg("in_array"), py::arg("out_array"), py::arg("shape"), py::arg("strides"), py::arg("offset"),
          "Element-wise set values from in_array to out_array (device-side)\n"
          "Args:\n"
          "  in_array: Input 1D CUDA Array (source values)\n"
          "  out_array: Output multi-dimensional CUDA Array (target)\n"
          "  shape: Int32Vector of out_array dimensions\n"
          "  strides: Int32Vector of out_array strides\n"
          "  offset: Starting offset in out_array");

    m.def("scalar_setitem", &ScalarSetitem, py::arg("size"), py::arg("value"), py::arg("out_array"), py::arg("shape"), py::arg("strides"), py::arg("offset"),
          "Set scalar value to multiple positions in out_array (device-side)\n"
          "Args:\n"
          "  size: Number of positions to set\n"
          "  value: Scalar float to set\n"
          "  out_array: Target CUDA Array\n"
          "  shape: Int32Vector of out_array dimensions\n"
          "  strides: Int32Vector of out_array strides\n"
          "  offset: Starting offset in out_array");

    // 绑定元素级操作函数（示例：ewise_add，其他类似）
    m.def("ewise_add", &EwiseAdd, py::arg("a_array"), py::arg("b_array"), py::arg("out_array"),
          "Element-wise addition of two CUDA arrays\n"
          "Args:\n"
          "  a_array: Input CUDA Array (float32)\n"
          "  b_array: Input CUDA Array (float32, same size as a_array)\n"
          "  out_array: Output CUDA Array (float32, same size as inputs)");

    m.def("scalar_add", &ScalarAdd, py::arg("a_array"), py::arg("value"), py::arg("out_array"),
          "Add scalar to each element of CUDA array\n"
          "Args:\n"
          "  a_array: Input CUDA Array (float32)\n"
          "  value: Scalar float to add\n"
          "  out_array: Output CUDA Array (same size as a_array)");

    // （注：其他函数 ewise_mul/scalar_mul/ewise_div 等均按上述格式添加文档字符串，此处省略重复）
    m.def("ewise_mul", &EwiseMul, py::arg("a_array"), py::arg("b_array"), py::arg("out_array"),
          "Element-wise multiplication of two CUDA arrays");
    m.def("scalar_mul", &ScalarMul, py::arg("a_array"), py::arg("value"), py::arg("out_array"),
          "Multiply each element of CUDA array by scalar");
    m.def("ewise_div", &EwiseDiv, py::arg("a_array"), py::arg("b_array"), py::arg("out_array"),
          "Element-wise division of two CUDA arrays");
    m.def("scalar_div", &ScalarDiv, py::arg("a_array"), py::arg("value"), py::arg("out_array"),
          "Divide each element of CUDA array by scalar (no division by zero)");
    m.def("scalar_power", &ScalarPower, py::arg("a_array"), py::arg("value"), py::arg("out_array"),
          "Raise each element of CUDA array to scalar power");
    m.def("ewise_maximum", &EwiseMaximum, py::arg("a_array"), py::arg("b_array"), py::arg("out_array"),
          "Element-wise maximum of two CUDA arrays");
    m.def("scalar_maximum", &ScalarMaximum, py::arg("a_array"), py::arg("value"), py::arg("out_array"),
          "Element-wise maximum of CUDA array and scalar");
    m.def("ewise_eq", &EwiseEq, py::arg("a_array"), py::arg("b_array"), py::arg("out_array"),
          "Element-wise equality check (returns 1.0 for equal, 0.0 otherwise)");
    m.def("scalar_eq", &ScalarEq, py::arg("a_array"), py::arg("value"), py::arg("out_array"),
          "Element-wise equality check with scalar (returns 1.0/0.0)");
    m.def("ewise_ge", &EwiseGe, py::arg("a_array"), py::arg("b_array"), py::arg("out_array"),
          "Element-wise >= check (returns 1.0/0.0)");
    m.def("scalar_ge", &ScalarGe, py::arg("a_array"), py::arg("value"), py::arg("out_array"),
          "Element-wise >= check with scalar (returns 1.0/0.0)");
    m.def("ewise_log", &EwiseLog, py::arg("in_array"), py::arg("out_array"),
          "Element-wise natural log (returns -INF for non-positive inputs)");
    m.def("ewise_exp", &EwiseExp, py::arg("in_array"), py::arg("out_array"),
          "Element-wise exponential (e^x)");
    m.def("ewise_tanh", &EwiseTanh, py::arg("in_array"), py::arg("out_array"),
          "Element-wise hyperbolic tangent");
    m.def("matmul", &Matmul, py::arg("a_array"), py::arg("b_array"), py::arg("out_array"), py::arg("M"), py::arg("N"), py::arg("P"),
          "Matrix multiplication: a(MxN) * b(NxP) = out(MxP)\n"
          "Args:\n"
          "  a_array: Input CUDA Array (MxN float32)\n"
          "  b_array: Input CUDA Array (NxP float32)\n"
          "  out_array: Output CUDA Array (MxP float32)\n"
          "  M: Rows of a and out\n"
          "  N: Columns of a, rows of b\n"
          "  P: Columns of b and out");
    m.def("reduce_max", &ReduceMax, py::arg("in_array"), py::arg("out_array"), py::arg("reduce_size"),
          "Reduce array by maximum: each out element = max of reduce_size consecutive in elements\n"
          "Args:\n"
          "  in_array: Input CUDA Array (size = out.size * reduce_size)\n"
          "  out_array: Output CUDA Array (size = in.size / reduce_size)\n"
          "  reduce_size: Number of elements to reduce per out element");
    m.def("reduce_sum", &ReduceSum, py::arg("in_array"), py::arg("out_array"), py::arg("reduce_size"),
          "Reduce array by sum: each out element = sum of reduce_size consecutive in elements");

    // 绑定数据传输函数：修复内存泄漏 + 文档说明
    m.def("to_numpy", [](const CudaArray& a, const std::vector<size_t>& shape, const std::vector<size_t>& strides, size_t offset) {
        std::vector<size_t> numpy_strides = strides;
        std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
            [](size_t& c) { return c * ELEM_SIZE; });

        // 修复内存泄漏：先分配，失败则释放
        scalar_t* host_ptr = new (std::nothrow) scalar_t[a.size];
        if (!host_ptr) {
            throw std::bad_alloc();
        }

        cudaError_t err = cudaMemcpy(host_ptr, a.get_ptr(), a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            delete[] host_ptr;  // 泄漏修复：拷贝失败时释放内存
            throw std::runtime_error("to_numpy: CUDA memcpy (device→host) failed: " + std::string(cudaGetErrorString(err)));
        }

        // 胶囊释放器：确保 numpy 数组生命周期结束后释放 host 内存
        py::capsule deallocate_buffer(host_ptr, [](void* p) {
            delete[] static_cast<scalar_t*>(p);
        });

        return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
    }, py::arg("cuda_array"), py::arg("shape"), py::arg("strides"), py::arg("offset"),
          "Transfer CUDA array to numpy array (device→host)\n"
          "Args:\n"
          "  cuda_array: Input CUDA Array\n"
          "  shape: SizeTVector of numpy array shape\n"
          "  strides: SizeTVector of numpy array strides (elements per step)\n"
          "  offset: Starting offset in CUDA array\n"
          "Returns:\n"
          "  numpy.ndarray (float32) with data from CUDA array");

    m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
        if (!out) throw std::invalid_argument("from_numpy: out_array cannot be null");
        if (a.size() != out->size) {
            // 修复核心问题：std::to_string 仅传入 out->size（数值类型），补充缺失的右引号和括号
            throw std::invalid_argument("from_numpy: numpy array size (" + std::to_string(a.size()) +
                                        ") mismatch with CUDA array size (" + std::to_string(out->size) + ")");
        }

        cudaError_t err = cudaMemcpy(out->get_ptr(), a.data(), out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("from_numpy: CUDA memcpy (host→device) failed: " + std::string(cudaGetErrorString(err)));
        }
    }, py::arg("numpy_array"), py::arg("out_array"),
          "Transfer numpy array to CUDA array (host→device)\n"
          "Args:\n"
          "  numpy_array: Input numpy.ndarray (float32)\n"
          "  out_array: Target CUDA Array (same size as numpy array)");
}