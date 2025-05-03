#pragma once

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include "storage/schema.hpp"
#include "storage/table.hpp"

namespace gpu_dbms {
namespace storage {

class CSVParser {
public:
    CSVParser();
    ~CSVParser();
    
    // Parse CSV file and extract schema
    std::shared_ptr<Schema> parseSchema(const std::string& csv_file, const std::string& table_name);
    
    // Parse CSV file and load data into memory
    std::shared_ptr<Table> parseData(const std::string& csv_file, const std::shared_ptr<Schema>& schema);
    
    // Write query results to CSV file
    bool writeToCSV(const std::string& csv_file, const std::shared_ptr<Table>& table);
    
private:
    // Split a line into fields, handling quotes and commas
    std::vector<std::string> splitLine(const std::string& line, char delimiter = ',');
    
    // Convert string to appropriate data type based on schema
    void convertData(const std::string& str, const DataType& type, ColumnData& col_data, size_t row);
};

} // namespace storage
} // namespace gpu_dbms