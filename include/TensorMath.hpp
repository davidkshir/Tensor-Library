#pragma once

#include "Tensor.hpp"
#include "TensorConcepts.hpp"
#include"details/TypePromotion.hpp"

namespace tensor {
    template<ReadableTensor T, NumericScalar Scalar>
    [[nodiscard]] Tensor<detail::PromoteType<typename T::valueType, Scalar>> add(const T& tensor, Scalar scalar);

    template<ReadableTensor T, ReadableTensor U>
    [[nodiscard]] Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> add(const T& tensor1, const U&  tensor2);

    template<ReadableTensor T, NumericScalar Scalar>
    [[nodiscard]] Tensor<detail::PromoteType<typename T::valueType, Scalar>> subtract(const T& tensor, Scalar scalar);

    template<ReadableTensor T, ReadableTensor U>
    [[nodiscard]] Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> subtract(const T& tensor1, const U&  tensor2);

    template<ReadableTensor T, NumericScalar Scalar>
    [[nodiscard]] Tensor<detail::PromoteType<typename T::valueType, Scalar>> multiply(const T& tensor, Scalar scalar);

    template<ReadableTensor T, ReadableTensor U>
    [[nodiscard]] Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> multiply(const T& tensor1, const U&  tensor2);

    template<ReadableTensor T, NumericScalar Scalar>
    [[nodiscard]] Tensor<detail::PromoteType<typename T::valueType, Scalar>> divide(const T& tensor, Scalar scalar);

    template<ReadableTensor T, ReadableTensor U>
    [[nodiscard]] Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> divide(const T& tensor1, const U&  tensor2);

    template<ReadableTensor T, ReadableTensor U>
    [[nodiscard]] Tensor<detail::PromoteType<typename T::valueType, typename U::valueType>> matMul(const T& tensor1, const U&  tensor2);



}
#include "TensorMath.tpp"