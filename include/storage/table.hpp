#pragma once

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include "storage/schema.hpp"

namespace gpu_dbms {
namespace storage {

// Column-oriented storage for better GPU processing
using IntColumn = std::vector<int>;
using FloatColumn = std::vector<double>;
using StringColumn = std::vector<std::string>;
using BoolColumn = std::vector<bool>;

// Variant to hold different column types
using ColumnData = std::variant<IntColumn, FloatColumn, StringColumn, BoolColumn>;

class Table {
public:
    Table(std::shared_ptr<Schema> schema);
    
    // Add a new column
    void addColumn(const std::string& column_name, ColumnData data);
    
    // Get column data
    const ColumnData& getColumn(const std::string& column_name) const;
    
    // Get schema
    std::shared_ptr<Schema> getSchema() const;
    
    // Get number of rows
    size_t numRows() const;
    
    // Get number of columns
    size_t numColumns() const;
    
    // Create empty table with same schema
    std::shared_ptr<Table> createEmptyTable() const;
    
private:
    std::shared_ptr<Schema> schema_;
    std::vector<ColumnData> columns_;
    std::unordered_map<std::string, size_t> column_map_; // Maps column name to index
    size_t num_rows_;
};

} // namespace storage
} // namespace gpu_dbms