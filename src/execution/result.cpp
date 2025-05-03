#include "execution/result.hpp"
#include <stdexcept>

namespace gpu_dbms {
namespace execution {

Result::Result(std::shared_ptr<storage::Schema> schema) : schema_(schema), data_(nullptr) {
    if (!schema_) {
        throw std::runtime_error("Result: Cannot create result with null schema");
    }
}

std::shared_ptr<storage::Schema> Result::getSchema() const {
    return schema_;
}

std::shared_ptr<storage::Table> Result::getData() const {
    return data_;
}

void Result::setData(std::shared_ptr<storage::Table> data) {
    if (data && data->getSchema() != schema_) {
        throw std::runtime_error("Result: Data table schema does not match result schema");
    }
    data_ = data;
}

size_t Result::numRows() const {
    return data_ ? data_->numRows() : 0;
}

} // namespace execution
} // namespace gpu_dbms