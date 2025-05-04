#pragma once

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include "storage/schema.hpp"

namespace gpu_dbms {
namespace storage {

using IntColumn = std::vector<int>;
using FloatColumn = std::vector<double>;
using StringColumn = std::vector<std::string>;
using BoolColumn = std::vector<bool>;

using ColumnData = std::variant<IntColumn, FloatColumn, StringColumn, BoolColumn>;

class Table {
public:
    Table(std::shared_ptr<Schema> schema);
    
    void addColumn(const std::string& column_name, ColumnData data);
    
    const ColumnData& getColumn(const std::string& column_name) const;
    ColumnData& getColumn(const std::string& column_name); // Added non-const overload
    
    std::shared_ptr<Schema> getSchema() const;
    
    size_t numRows() const;
    
    size_t numColumns() const;
    
    std::shared_ptr<Table> createEmptyTable() const;
    
    bool hasColumn(const std::string& column_name) const; // Added hasColumn

private:
    std::shared_ptr<Schema> schema_;
    std::vector<ColumnData> columns_;
    std::unordered_map<std::string, size_t> column_map_;
    size_t num_rows_;
};

} // namespace storage
} // namespace gpu_dbms