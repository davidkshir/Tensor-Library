#pragma once

#include <vector>
#include <cstddef>

namespace tensor {

    template<typename T>

    class Storage {
    public:
        T& operator[](std::size_t index);

        const T& operator[](std::size_t index) const;


    private:
        std::vector<T> storage;
    };
}
#include "Storage.tpp"