#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  size_t cnt = 0;
  
  // Calculate total number of elements
  size_t total_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    total_size *= shape[i];
  }
  
  // Use a vector to maintain indices for each dimension
  std::vector<size_t> indices(ndim, 0);
  
  // Iterate through all elements
  for (size_t i = 0; i < total_size; i++) {
    // Calculate the offset in the input array
    size_t input_idx = offset;
    for (size_t dim = 0; dim < ndim; dim++) {
      input_idx += indices[dim] * strides[dim];
    }
    
    // Copy element from input to output
    out->ptr[cnt++] = a.ptr[input_idx];
    
    // Increment indices (like a counter with carry)
    size_t dim = ndim - 1;
    while (dim >= 0) {
      indices[dim]++;
      if (indices[dim] < shape[dim]) {
        break;  // No carry needed
      }
      indices[dim] = 0;  // Reset this dimension and carry
      if (dim == 0) break;  // Prevent underflow when dim is size_t
      dim--;
    }
  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  size_t cnt = 0;
  
  // Calculate total number of elements
  size_t total_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    total_size *= shape[i];
  }
  
  // Use a vector to maintain indices for each dimension
  std::vector<size_t> indices(ndim, 0);
  
  // Iterate through all elements
  for (size_t i = 0; i < total_size; i++) {
    // Calculate the offset in the output array
    size_t output_idx = offset;
    for (size_t dim = 0; dim < ndim; dim++) {
      output_idx += indices[dim] * strides[dim];
    }
    
    // Copy element from compact input to strided output
    out->ptr[output_idx] = a.ptr[cnt++];
    
    // Increment indices (like a counter with carry)
    size_t dim = ndim - 1;
    while (dim >= 0) {
      indices[dim]++;
      if (indices[dim] < shape[dim]) {
        break;  // No carry needed
      }
      indices[dim] = 0;  // Reset this dimension and carry
      if (dim == 0) break;  // Prevent underflow when dim is size_t
      dim--;
    }
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  size_t ndim = shape.size();
  
  // Use a vector to maintain indices for each dimension
  std::vector<size_t> indices(ndim, 0);
  
  // Iterate through all elements
  for (size_t i = 0; i < size; i++) {
    // Calculate the offset in the output array
    size_t output_idx = offset;
    for (size_t dim = 0; dim < ndim; dim++) {
      output_idx += indices[dim] * strides[dim];
    }
    
    // Set the element to the scalar value
    out->ptr[output_idx] = val;
    
    // Increment indices (like a counter with carry)
    size_t dim = ndim - 1;
    while (dim >= 0) {
      indices[dim]++;
      if (indices[dim] < shape[dim]) {
        break;  // No carry needed
      }
      indices[dim] = 0;  // Reset this dimension and carry
      if (dim == 0) break;  // Prevent underflow when dim is size_t
      dim--;
    }
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the product of corresponding entries in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the product of corresponding entry in a and the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  }
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the division of corresponding entries in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / b.ptr[i];
  }
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the division of corresponding entry in a and the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] / val;
  }
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the power of corresponding entry in a to the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the maximum of corresponding entries in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the maximum of corresponding entry in a and the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to 1.0 if corresponding entries in a and b are equal, 0.0 otherwise.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == b.ptr[i]) ? 1.0f : 0.0f;
  }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to 1.0 if corresponding entry in a equals the scalar val, 0.0 otherwise.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] == val) ? 1.0f : 0.0f;
  }
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to 1.0 if corresponding entry in a >= corresponding entry in b, 0.0 otherwise.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= b.ptr[i]) ? 1.0f : 0.0f;
  }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to 1.0 if corresponding entry in a >= the scalar val, 0.0 otherwise.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = (a.ptr[i] >= val) ? 1.0f : 0.0f;
  }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the natural logarithm of corresponding entries in a.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the exponential of corresponding entries in a.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  /**
   * Set entries in out to be the hyperbolic tangent of corresponding entries in a.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  // Initialize output to zero
  for (size_t i = 0; i < m * p; i++) {
    out->ptr[i] = 0;
  }
  
  // Naive three-loop matrix multiplication
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < p; j++) {
      for (uint32_t k = 0; k < n; k++) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  // Multiply two TILE x TILE matrices and ADD to out
  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  uint32_t m_tiles = m / TILE;
  uint32_t n_tiles = n / TILE;
  uint32_t p_tiles = p / TILE;
  
  // Initialize output to zero
  for (size_t i = 0; i < out->size; i++) {
    out->ptr[i] = 0;
  }
  
  // Iterate over output tiles
  for (uint32_t i = 0; i < m_tiles; i++) {
    for (uint32_t j = 0; j < p_tiles; j++) {
      // For each output tile (i,j), sum over all k
      for (uint32_t k = 0; k < n_tiles; k++) {
        // Calculate pointers to the tiles
        const float* a_tile = a.ptr + (i * n_tiles + k) * TILE * TILE;
        const float* b_tile = b.ptr + (k * p_tiles + j) * TILE * TILE;
        float* out_tile = out->ptr + (i * p_tiles + j) * TILE * TILE;
        
        // Call AlignedDot to multiply and add tiles
        AlignedDot(a_tile, b_tile, out_tile);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    // Initialize with first element of the block
    scalar_t max_val = a.ptr[i * reduce_size];
    
    // Find maximum over the block
    for (size_t j = 1; j < reduce_size; j++) {
      scalar_t val = a.ptr[i * reduce_size + j];
      if (val > max_val) {
        max_val = val;
      }
    }
    
    out->ptr[i] = max_val;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    // Initialize sum to zero
    scalar_t sum = 0;
    
    // Sum over the block
    for (size_t j = 0; j < reduce_size; j++) {
      sum += a.ptr[i * reduce_size + j];
    }
    
    out->ptr[i] = sum;
  }
  /// END SOLUTION
}

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
