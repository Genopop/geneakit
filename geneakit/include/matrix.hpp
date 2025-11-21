#ifndef MATRIX_H
#define MATRIX_H
#include <cstddef>
#include <cstring>
#include <iostream>
#include <memory>

template <typename T>
class Matrix {
public:
    // Constructor
    Matrix(size_t rows, size_t cols) :
        m_rows(rows), m_cols(cols),
        m_data(std::make_unique<T[]>(rows * cols)) {}
    
    // Accessors
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    size_t size() const { 
        if (m_rows == m_cols) return m_rows;
        else return std::cerr << "Matrix is not square." << std::endl, 0; 
    }
    
    T* data() { return m_data.get(); }
    const T* data() const { return m_data.get(); }
    
    // Copy constructor
    Matrix(const Matrix& other) :
        m_rows(other.m_rows), m_cols(other.m_cols),
        m_data(std::make_unique<T[]>(other.m_rows * other.m_cols)) {
        std::copy(other.m_data.get(), other.m_data.get() + m_rows * m_cols, 
                  m_data.get());
    }
    
    // Copy assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;
        m_data = std::make_unique<T[]>(other.m_rows * other.m_cols);
        std::copy(other.m_data.get(), other.m_data.get() + 
                  other.m_rows * other.m_cols, m_data.get());
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        return *this;
    }
    
    // Move constructor
    Matrix(Matrix&& other) noexcept :
        m_rows(other.m_rows), m_cols(other.m_cols),
        m_data(std::move(other.m_data)) {
        other.m_rows = 0;
        other.m_cols = 0;
    }
    
    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this == &other) return *this;
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_data = std::move(other.m_data);
        other.m_rows = 0;
        other.m_cols = 0;
        return *this;
    }
    
    // Clear
    void clear() {
        m_data.reset();
        m_rows = 0;
        m_cols = 0;
    }
    
    // Subscript operator
    T* operator[](size_t row) { return &m_data[row * m_cols]; }
    const T* operator[](size_t row) const { return &m_data[row * m_cols]; }

private:
    size_t m_rows, m_cols;
    std::unique_ptr<T[]> m_data;
};

template <typename T>
Matrix<T> zeros(size_t rows, size_t cols) {
    Matrix<T> matrix(rows, cols);
    std::memset(matrix.data(), 0, rows * cols * sizeof(T));  // Fixed typo: std::memset
    return matrix;
}

#endif