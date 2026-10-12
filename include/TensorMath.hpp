#pragma once

#include "Tensor.hpp"
#include "TensorConcepts.hpp"
#include"details/TypePromotion.hpp"

namespace tensor {
    template<typename T, NumericScalar Scalar>
    [[nodiscard]] Tensor<detail::PromoteType<T, Scalar>> add(const Tensor<T>& tensor, Scalar scalar);

    template<typename T, typename U>
    [[nodiscard]] Tensor<detail::PromoteType<T, U>> add(const Tensor<T>& tensor1, const Tensor<U>&  tensor2);

    template<typename T, NumericScalar Scalar>
    [[nodiscard]] Tensor<detail::PromoteType<T, Scalar>> subtract(const Tensor<T>& tensor, Scalar scalar);

    template<typename T, typename U>
    [[nodiscard]] Tensor<detail::PromoteType<T, U>> subtract(const Tensor<T>& tensor1, const Tensor<U>&  tensor2);

    template<typename T, NumericScalar Scalar>
    [[nodiscard]] Tensor<detail::PromoteType<T, Scalar>> multiply(const Tensor<T>& tensor, Scalar scalar);

    template<typename T, typename U>
    [[nodiscard]] Tensor<detail::PromoteType<T, U>> multiply(const Tensor<T>& tensor1, const Tensor<U>&  tensor2);

    template<typename T, NumericScalar Scalar>
    [[nodiscard]] Tensor<detail::PromoteType<T, Scalar>> divide(const Tensor<T>& tensor, Scalar scalar);

    template<typename T, typename U>
    [[nodiscard]] Tensor<detail::PromoteType<T, U>> divide(const Tensor<T>& tensor1, const Tensor<U>&  tensor2);

    template<typename T, typename U>
    [[nodiscard]] Tensor<detail::PromoteType<T, U>> matMul(const Tensor<T>& tensor1, const Tensor<U>&  tensor2);



}
#include "TensorMath.tpp"