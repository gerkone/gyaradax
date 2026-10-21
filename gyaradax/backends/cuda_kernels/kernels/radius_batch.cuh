#pragma once

#include <cstdint>

// expand_dims vmap: leading axis of size n per vmapped element, or 1 when the operand is shared

template <class Buf>
inline int64_t batch_stride_bytes(const Buf& b, int64_t n) {
    const auto d = b.dimensions();
    return (n > 1 && d.size() > 0 && d[0] == n) ? static_cast<int64_t>(b.size_bytes()) / n : 0;
}

template <class T, class Buf>
inline T* batch_at(const Buf& b, int64_t n, int64_t i) {
    return reinterpret_cast<T*>(static_cast<char*>(b.untyped_data()) + batch_stride_bytes(b, n) * i);
}

// number of vmapped elements, from an output of `item` elements per element
template <class Buf>
inline int64_t batch_count(const Buf& out, int64_t item) {
    return item > 0 ? static_cast<int64_t>(out.element_count()) / item : 1;
}
