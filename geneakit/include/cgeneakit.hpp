#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>

/*------------------------------------------------------------------------------
MIT License

Copyright (c) 2024 Gilles-Philippe Morin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------*/

#include "create.hpp"
#include "identify.hpp"
#include "extract.hpp"
#include "output.hpp"
#include "describe.hpp"
#include "compute.hpp"
#include "edit.hpp"

namespace nb = nanobind;

// Walks the individuals of a pedigree in `ids` order, which is the same
// order as `keys()` and `__iter__`. Iterating the `individuals` map
// directly would be a different, arbitrary order, so `values()` and
// `items()` would not line up with `keys()`.
template <typename Element>
class PedigreeIterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Element;
    using difference_type = std::ptrdiff_t;
    using pointer = const Element *;
    using reference = Element;

    PedigreeIterator(Pedigree<> *pedigree, std::size_t index) :
        pedigree(pedigree), index(index) {}

    Element operator*() const;

    PedigreeIterator &operator++() {
        index++;
        return *this;
    }
    bool operator==(const PedigreeIterator &other) const {
        return index == other.index;
    }
    bool operator!=(const PedigreeIterator &other) const {
        return index != other.index;
    }

private:
    Pedigree<> *pedigree;
    std::size_t index;
};

template <>
inline Individual<> *PedigreeIterator<Individual<> *>::operator*() const {
    return pedigree->individuals.at(pedigree->ids[index]);
}

template <>
inline std::pair<int, Individual<> *>
PedigreeIterator<std::pair<int, Individual<> *>>::operator*() const {
    const int id = pedigree->ids[index];
    return {id, pedigree->individuals.at(id)};
}

using PedigreeValueIterator = PedigreeIterator<Individual<> *>;
using PedigreeItemIterator =
    PedigreeIterator<std::pair<int, Individual<> *>>;