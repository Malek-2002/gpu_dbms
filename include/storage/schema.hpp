#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace gpu_dbms {
namespace storage {

enum class DataType {
    INT,
    FLOAT,
    STRING,
    BOOLEAN
};

struct ColumnInfo {
    std::string name;
    DataType type;
    bool is_primary_key;
    bool is_foreign_key;
    std::string referenced_table;  // If foreign key
    std::string referenced_column; // If foreign key
};

class Schema {
public:
    Schema(const std::string& table_name);
    
    void addColumn(const ColumnInfo& column_info);
    const ColumnInfo& getColumn(const std::string& column_name) const;
    const std::vector<ColumnInfo>& getColumns() const;
    const std::string& getTableName() const;
    
    // Check if a column exists
    bool hasColumn(const std::string& column_name) const;
    
    // Get primary key column(s)
    std::vector<std::string> getPrimaryKeyColumns() const;
    
    // Get foreign key relationships
    std::vector<std::pair<std::string, std::string>> getForeignKeyRelationships() const;
    
private:
    std::string table_name_;
    std::vector<ColumnInfo> columns_;
    std::unordered_map<std::string, size_t> column_map_; // Maps column name to index
};

} // namespace storage
} // namespace gpu_dbms