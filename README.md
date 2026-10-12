# TensorLib

An N-dimensional tensor library written from scratch in modern C++20. The project implements tensor storage, multidimensional indexing, views, broadcasting, and numerical operations without external numerical computing or machine learning libraries.

TensorLib is designed as a foundation for exploring the underlying implementation of tensor computation and, eventually, automatic differentiation.

## Features

- N-dimensional tensors with configurable shapes
- Contiguous storage with stride-based multidimensional indexing
- Noncontiguous tensor views with shared underlying storage
- Mutable and immutable tensor views
- Slicing with configurable start, end, and step
- Reshaping, transposing, and axis permutation
- Deep copying through `clone()`
- Broadcasting
- Elementwise addition, subtraction, multiplication, and division
- Tensor–scalar and tensor–tensor arithmetic
- Automatic numeric type promotion for mixed-type operations
- Matrix multiplication with batched broadcasting
- Support for mathematical operations on noncontiguous views
- C++20 concepts for generic tensor operations
- No external numerical computing dependencies

## Implementation

The library is organized around four primary abstractions:

**Storage → TensorLayout → Tensor → TensorView / MutableTensorView**

**Storage:** Manages the underlying contiguous collection of tensor elements.

**TensorLayout:** Describes how tensor elements are interpreted in memory through shape, strides, and storage offset. Handles multidimensional indexing, slicing, reshaping, transposition, axis permutation, and broadcasting.

**Tensor:** Owns a tensor layout and shared storage. Provides element access, mathematical operations through the library's math API, and methods for creating views.

**TensorView / MutableTensorView:** Provide read-only or mutable access to tensor data through independent layouts that share the underlying storage. This allows operations such as slicing and transposition without copying the tensor's elements.

### Memory Layout

TensorLib separates logical tensor dimensions from physical storage.

Each tensor layout maintains:

- **Shape:** The size of each dimension
- **Strides:** The number of storage positions between elements along each dimension
- **Offset:** The starting position within the underlying storage

A multidimensional index is converted into a storage index using:

`storageIndex = offset + Σ(indices[i] * strides[i])`

This representation allows noncontiguous tensors, transposed views, and broadcasted dimensions to be accessed using the same underlying storage.

### Broadcasting

TensorLib supports NumPy-style broadcasting for compatible tensor shapes.

Dimensions are compared from right to left. Two dimensions are compatible when they are equal or one of them is `1`.

For example:

```text
Tensor A:   [3, 1, 4]
Tensor B:   [1, 5, 1]
Result:     [3, 5, 4]
```

Broadcasted dimensions use zero strides, allowing repeated logical access to existing elements without duplicating the input data.

### Mathematical Operations

TensorLib currently provides:

| Operation | Function |
|---|---|
| Addition | `add()` |
| Subtraction | `subtract()` |
| Multiplication | `multiply()` |
| Division | `divide()` |
| Matrix multiplication | `matMul()` |

Elementwise arithmetic supports tensor–scalar and tensor–tensor operations, including broadcasting and mixed numeric types.

Matrix multiplication supports two-dimensional matrices and higher-dimensional batched inputs, including compatible batch broadcasting.

Mathematical functions use the `ReadableTensor` concept to accept owning tensors and tensor views through a common interface.

## Usage Example

### Creating Tensors and Performing Arithmetic

```cpp
#include <iostream>
#include "Tensor.hpp"
#include "TensorMath.hpp"

int main() {
    using namespace tensor;

    Tensor<float> a({2, 2}, {
        1.0f, 2.0f,
        3.0f, 4.0f
    });

    Tensor<float> b({2, 2}, {
        5.0f, 6.0f,
        7.0f, 8.0f
    });

    auto sum = add(a, b);
    auto product = matMul(a, b);

    std::cout << sum.at({0, 0}) << '\n';     // 6
    std::cout << product.at({0, 0}) << '\n'; // 19
}
```

### Noncontiguous Tensor Views

Tensor views allow operations on selected elements without copying the original storage.

```cpp
#include "Tensor.hpp"
#include "TensorMath.hpp"

int main() {
    using namespace tensor;

    Tensor<int> a({2, 4}, {
        1, 2, 3, 4,
        5, 6, 7, 8
    });

    // Select every second column.
    auto view = a.slice({Slice{1, 0, 4, 2}});

    // Logical view:
    // [1, 3]
    // [5, 7]

    auto result = multiply(view, 2);

    // Result:
    // [2,  6]
    // [10, 14]
}
```

The view maintains its own shape, strides, and offset while referencing the original tensor's storage. Mathematical operations respect these layouts when accessing elements.



## Building

TensorLib uses C++20 and CMake.

From the project's root directory:

```bash
cmake -S . -B build
cmake --build build
```

The library's public headers are located in `include/`, with supporting implementation files in `src/`.

to extend the library into an automatic differentiation framework capable of supporting neural network development while maintaining control over tensor storage, memory layouts, and mathematical operations.