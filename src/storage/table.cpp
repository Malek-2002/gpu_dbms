#include "storage/table.hpp"
#include <stdexcept>

namespace gpu_dbms {
namespace storage {

Table::Table(std::shared_ptr<Schema> schema) 
    : schema_(schema), num_rows_(0) {
    // Pre-allocate columns based on schema
    const auto& schema_columns = schema->getColumns();
    columns_.resize(schema_columns.size());
    
    for (size_t i = 0; i < schema_columns.size(); ++i) {
        const auto& col_info = schema_columns[i];
        column_map_[col_info.name] = i;
        
        // Initialize empty columns with proper type
        switch (col_info.type) {
            case DataType::INT:
                columns_[i] = IntColumn();
                break;
            case DataType::FLOAT:
                columns_[i] = FloatColumn();
                break;
            case DataType::STRING:
                columns_[i] = StringColumn();
                break;
            case DataType::BOOLEAN:
                columns_[i] = BoolColumn();
                break;
        }
    }
}

void Table::addColumn(const std::string& column_name, ColumnData data) {
    auto it = column_map_.find(column_name);
    if (it == column_map_.end()) {
        throw std::runtime_error("Column " + column_name + " does not exist in schema");
    }
    
    size_t data_size = 0;
    std::visit([&](const auto& col) { data_size = col.size(); }, data);
    
    if (num_rows_ == 0) {
        // First column being added, set num_rows_
        num_rows_ = data_size;
    } else if (data_size != num_rows_) {
        throw std::runtime_error("Column " + column_name + " has incorrect number of rows");
    }
    
    columns_[it->second] = std::move(data);
}

const ColumnData& Table::getColumn(const std::string& column_name) const {
    auto it = column_map_.find(column_name);
    if (it == column_map_.end()) {
        throw std::runtime_error("Column " + column_name + " does not exist");
    }
    
    return columns_[it->second];
}

std::shared_ptr<Schema> Table::getSchema() const {
    return schema_;
}

size_t Table::numRows() const {
    return num_rows_;
}

size_t Table::numColumns() const {
    return columns_.size();
}

std::shared_ptr<Table> Table::createEmptyTable() const {
    return std::make_shared<Table>(schema_);
}

} // namespace storage
} // namespace gpu_dbms