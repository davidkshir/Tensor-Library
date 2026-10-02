#include <stdexcept>
#include <utility>
#include "TensorLayout.hpp"

#include "details/MathHelpers.hpp"

namespace tensor::detail {
    std::pair<TensorLayout, TensorLayout> broadcast(const TensorLayout &a, const TensorLayout &b) {
        const Shape& aShape = a.getShape();
        const Shape& bShape = b.getShape();

        const std::size_t rank =  std::max(aShape.size(), bShape.size());

        const std::size_t aOffset = rank - aShape.size();
        const std::size_t bOffset = rank - bShape.size();

        Shape commonShape(rank);

        for (std::size_t i = 0; i < rank; ++i) {
            std::size_t aDim;
            std::size_t bDim;

            // Find aDim
            if (i < aOffset) {
                aDim = 1;
            }
            else {
                aDim = aShape[i - aOffset];
            }

            // Find bDim
            if (i < bOffset) {
                bDim = 1;
            }
            else {
                bDim = bShape[i - bOffset];
            }

            // Compare dims
            if (aDim == bDim) {
                commonShape[i] = aDim; // Can choose either dim
            }
            else if (aDim == 1) {
                commonShape[i] = bDim;
            }
            else if (bDim == 1) {
                commonShape[i] = aDim;
            }
            else {
                throw std::invalid_argument("Incompatible shapes to broadcast.");
            }
        }
        return {a.broadcastTo(commonShape), b.broadcastTo(commonShape)};
    }
}
