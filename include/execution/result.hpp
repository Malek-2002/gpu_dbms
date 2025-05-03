#pragma once

#include <memory>
#include <vector>
#include "storage/schema.hpp"
#include "storage/table.hpp"

namespace gpu_dbms {
namespace execution {

class Result {
public:
    Result(std::shared_ptr<storage::Schema> schema);
    
    // Get the schema of the result
    std::shared_ptr<storage::Schema> getSchema() const;
    
    // Get the data as a table
    std::shared_ptr<storage::Table> getData() const;
    
    // Set the data
    void setData(std::shared_ptr<storage::Table> data);
    
    // Get number of rows
    size_t numRows() const;

private:
    std::shared_ptr<storage::Schema> schema_;
    std::shared_ptr<storage::Table> data_;
};

} // namespace execution
} // namespace gpu_dbms