#include "storage/catalog.hpp"
#include "storage/csv_parser.hpp"
#include <stdexcept>

namespace gpu_dbms {
namespace storage {

Catalog& Catalog::getInstance() {
    static Catalog instance;
    return instance;
}

void Catalog::addSchema(std::shared_ptr<Schema> schema) {
    if (!schema) {
        throw std::invalid_argument("Schema cannot be null");
    }
    
    const std::string& table_name = schema->getTableName();
    schemas_[table_name] = schema;
}

std::shared_ptr<Schema> Catalog::getSchema(const std::string& table_name) const {
    auto it = schemas_.find(table_name);
    if (it == schemas_.end()) {
        throw std::runtime_error("Schema not found for table: " + table_name);
    }
    return it->second;
}

bool Catalog::hasTable(const std::string& table_name) const {
    return tables_.find(table_name) != tables_.end();
}

void Catalog::addTable(std::shared_ptr<Table> table) {
    if (!table) {
        throw std::invalid_argument("Table cannot be null");
    }
    
    const std::string table_name = table->getSchema()->getTableName();
    tables_[table_name] = table;
    
    // Ensure schema is also added
    schemas_[table_name] = table->getSchema();
}

std::shared_ptr<Table> Catalog::getTable(const std::string& table_name) const {
    auto it = tables_.find(table_name);
    if (it == tables_.end()) { // Fixed: Check tables_.end() instead of schemas_.end()
        throw std::runtime_error("Table not found: " + table_name);
    }
    return it->second;
}

void Catalog::loadFromCSV(const std::string& table_name, const std::string& csv_file) {
    
    // Check if schema exists; if not, infer from CSV
    std::shared_ptr<Schema> schema;
    auto schema_it = schemas_.find(table_name);
    if (schema_it == schemas_.end()) {
        CSVParser parser;
        schema = parser.parseSchema(csv_file, table_name);
        if (!schema) {
            throw std::runtime_error("Failed to create schema from CSV file: " + csv_file);
        }
        schemas_[table_name] = schema;
    } else {
        schema = schema_it->second;
    }
    
    // Parse CSV file into a table
    CSVParser parser;
    auto table = parser.parseData(csv_file, schema);
    if (!table) {
        throw std::runtime_error("Failed to load CSV file: " + csv_file);
    }
    
    // Add table to catalog using addTable to ensure schema consistency
    addTable(table);
}

std::vector<std::string> Catalog::getAllTableNames() const {
    std::vector<std::string> names;
    names.reserve(tables_.size());
    for (const auto& pair : tables_) {
        names.push_back(pair.first);
    }
    return names;
}

} // namespace storage
} // namespace gpu_dbms