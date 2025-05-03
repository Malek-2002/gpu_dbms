#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include "storage/schema.hpp"
#include "storage/table.hpp"

namespace gpu_dbms {
namespace storage {

class Catalog {
public:
    static Catalog& getInstance();
    
    // Add schema to catalog
    void addSchema(std::shared_ptr<Schema> schema);
    
    // Get schema by table name
    std::shared_ptr<Schema> getSchema(const std::string& table_name) const;
    
    // Check if table exists
    bool hasTable(const std::string& table_name) const;
    
    // Add table to catalog
    void addTable(std::shared_ptr<Table> table);
    
    // Get table by name
    std::shared_ptr<Table> getTable(const std::string& table_name) const;
    
    // Load data from CSV file
    void loadFromCSV(const std::string& table_name, const std::string& csv_file);
    
    // Get all table names
    std::vector<std::string> getAllTableNames() const;
    
private:
    Catalog() = default;
    ~Catalog() = default;
    
    Catalog(const Catalog&) = delete;
    Catalog& operator=(const Catalog&) = delete;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<Schema>> schemas_;
    std::unordered_map<std::string, std::shared_ptr<Table>> tables_;
};

} // namespace storage
} // namespace gpu_dbms