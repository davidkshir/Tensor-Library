#include "details/TensorHelpers.hpp"
#include <stdexcept>

namespace tensor::detail {

    std::size_t advanceIndices(Indices& indices, const Shape& shape) { // Expects indices filled with 0s for the size of dimensions
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("Indices and shape must have the same dimensions.");
        }

        for (std::size_t axis = shape.size(); axis-- > 0;) {
            indices[axis]++;

            if (indices[axis] < shape[axis]) {
                return axis; // Axis that successfully advanced
            }

            indices[axis] = 0; // This axis rolled over
        }

        return shape.size(); // No valid axis advanced therefore traversal is complete
    }

}