#pragma once

#include <cstddef>
#include <vector>

namespace tensor {
    using Strides = std::vector<std::size_t>;
    using Shape = std::vector<std::size_t>;
    using Indices = std::vector<std::size_t>;

    struct Slice {
        std::size_t axis{};
        std::size_t start{};
        std::size_t end{};
        std::size_t step = 1;
    };

    using Slices = std::vector<Slice>;
}
