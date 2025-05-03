#include "storage/schema.hpp"
#include <stdexcept>

namespace gpu_dbms {
namespace storage {

Schema::Schema(const std::string& table_name) : table_name_(table_name) {}

void Schema::addColumn(const ColumnInfo& column_info) {
    if (column_map_.find(column_info.name) != column_map_.end()) {
        throw std::runtime_error("Column " + column_info.name + " already exists in table " + table_name_);
    }
    
    column_map_[column_info.name] = columns_.size();
    columns_.push_back(column_info);
}

const ColumnInfo& Schema::getColumn(const std::string& column_name) const {
    auto it = column_map_.find(column_name);
    if (it == column_map_.end()) {
        throw std::runtime_error("Column " + column_name + " does not exist in table " + table_name_);
    }
    
    return columns_[it->second];
}

const std::vector<ColumnInfo>& Schema::getColumns() const {
    return columns_;
}

const std::string& Schema::getTableName() const {
    return table_name_;
}

std::vector<std::string> Schema::getPrimaryKeyColumns() const {
    std::vector<std::string> pk_columns;
    for (const auto& column : columns_) {
        if (column.is_primary_key) {
            pk_columns.push_back(column.name);
        }
    }
    return pk_columns;
}

std::vector<std::pair<std::string, std::string>> Schema::getForeignKeyRelationships() const {
    std::vector<std::pair<std::string, std::string>> fk_relationships;
    for (const auto& column : columns_) {
        if (column.is_foreign_key) {
            fk_relationships.emplace_back(column.name, 
                                          column.referenced_table + "." + column.referenced_column);
        }
    }
    return fk_relationships;
}

} // namespace storage
} // namespace gpu_dbms