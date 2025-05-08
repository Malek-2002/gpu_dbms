#include "storage/csv_parser.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>

namespace gpu_dbms {
namespace storage {

CSVParser::CSVParser() = default;
CSVParser::~CSVParser() = default;

std::shared_ptr<Schema> CSVParser::parseSchema(const std::string& csv_file, const std::string& table_name) {
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + csv_file);
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("Empty CSV file: " + csv_file);
    }

    auto column_names = splitLine(header_line);
    auto schema = std::make_shared<Schema>(table_name);

    auto trim = [](std::string& s) {
        s.erase(0, s.find_first_not_of(" \t\n\r\f\v"));
        s.erase(s.find_last_not_of(" \t\n\r\f\v") + 1);
    };

    for (auto& col_raw : column_names) {
        std::string col_name = col_raw;
        bool is_primary_key = false;
        bool is_foreign_key = false;
        std::string referenced_table;
        std::string referenced_column;
        DataType type = DataType::STRING;  // Default type

        // Tag extractor and eraser
        auto extractAndEraseTag = [&](const std::string& tag) {
            size_t pos = col_name.find(tag);
            if (pos != std::string::npos) {
                col_name.erase(pos, tag.length());
                return true;
            }
            return false;
        };

        // Extract known tags
        if (extractAndEraseTag("(P)")) is_primary_key = true;
        if (extractAndEraseTag("(F)")) is_foreign_key = true;
        if (extractAndEraseTag("(T)")) type = DataType::STRING;
        if (extractAndEraseTag("(D)")) type = DataType::STRING; // Can replace with DataType::DATE if supported
        if (extractAndEraseTag("(N)")) type = DataType::FLOAT;

        // Trim remaining spaces
        trim(col_name);

        // Check for foreign key by #ReferencedTable_column pattern
        if (!col_name.empty() && col_name.front() == '#') {
            is_foreign_key = true;
            size_t underscore_pos = col_name.find_last_of('_');
            if (underscore_pos != std::string::npos) {
                referenced_table = col_name.substr(1, underscore_pos - 1);
                referenced_column = col_name.substr(underscore_pos + 1);
                col_name = col_name.substr(1);  // remove '#'
            }
        }

        ColumnInfo col_info{
            col_name,
            type,
            is_primary_key,
            is_foreign_key,
            referenced_table,
            referenced_column
        };

        schema->addColumn(col_info);
    }

    return schema;
}

std::shared_ptr<Table> CSVParser::parseData(const std::string& csv_file, const std::shared_ptr<Schema>& schema) {
    std::ifstream file(csv_file);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + csv_file);
    }
    
    auto table = std::make_shared<Table>(schema);
    const auto& columns = schema->getColumns();
    
    // Skip header
    std::string line;
    std::getline(file, line);
    
    // Prepare column data containers
    std::vector<ColumnData> column_data(columns.size());
    for (size_t i = 0; i < columns.size(); ++i) {
        switch (columns[i].type) {
            case DataType::INT:
                column_data[i] = IntColumn();
                break;
            case DataType::FLOAT:
                column_data[i] = FloatColumn();
                break;
            case DataType::STRING:
                column_data[i] = StringColumn();
                break;
            case DataType::BOOLEAN:
                column_data[i] = BoolColumn();
                break;
        }
    }
    
    // Read data rows
    size_t row_index = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        auto values = splitLine(line);
        if (values.size() != columns.size()) {
            throw std::runtime_error("Invalid row format at line " + std::to_string(row_index + 2));
        }
        
        // Add row data to each column
        for (size_t i = 0; i < columns.size(); ++i) {
            convertData(values[i], columns[i].type, column_data[i], row_index);
        }
        
        row_index++;
    }
    
    // Add columns to table
    for (size_t i = 0; i < columns.size(); ++i) {
        table->addColumn(columns[i].name, column_data[i]);
    }
    
    return table;
}

bool CSVParser::writeToCSV(const std::string& csv_file, const std::shared_ptr<Table>& table) {
    std::ofstream file(csv_file);
    if (!file.is_open()) {
        return false;
    }
    
    const auto& schema = table->getSchema();
    const auto& columns = schema->getColumns();
    const size_t num_rows = table->numRows();
    
    // Write header
    for (size_t i = 0; i < columns.size(); ++i) {
        file << columns[i].name;
        if (i < columns.size() - 1) {
            file << ",";
        }
    }
    file << "\n";
    
    // Write data rows
    for (size_t row = 0; row < num_rows; ++row) {
        for (size_t i = 0; i < columns.size(); ++i) {
            const auto& col_data = table->getColumn(columns[i].name);
            
            std::visit([&](const auto& data) {
                using T = std::decay_t<decltype(data)>;
                if constexpr (std::is_same_v<T, StringColumn>) {
                    // Quote strings if they contain commas
                    const auto& value = data[row];
                    if (value.find(',') != std::string::npos) {
                        file << "\"" << value << "\"";
                    } else {
                        file << value;
                    }
                } else if constexpr (std::is_same_v<T, BoolColumn>) {
                    file << (data[row] ? "true" : "false");
                } else {
                    file << data[row];
                }
            }, col_data);
            
            if (i < columns.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    
    return true;
}

std::vector<std::string> CSVParser::splitLine(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::string field;
    bool in_quotes = false;
    
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == delimiter && !in_quotes) {
            // Trim trailing whitespace and \r from field
            field.erase(std::find_if(field.rbegin(), field.rend(), [](unsigned char ch) {
                return !std::isspace(ch) && ch != '\r';
            }).base(), field.end());
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    
    // Trim the last field
    field.erase(std::find_if(field.rbegin(), field.rend(), [](unsigned char ch) {
        return !std::isspace(ch) && ch != '\r';
    }).base(), field.end());
    fields.push_back(field);
    
    return fields;
}

void CSVParser::convertData(const std::string& str, const DataType& type, ColumnData& col_data, size_t row) {
    switch (type) {
        case DataType::INT: {
            auto& int_col = std::get<IntColumn>(col_data);
            if (row >= int_col.size()) {
                int_col.resize(row + 1);
            }
            try {
                int_col[row] = std::stoi(str);
            } catch (...) {
                int_col[row] = 0; // Default value for invalid conversion
            }
            break;
        }
        case DataType::FLOAT: {
            auto& float_col = std::get<FloatColumn>(col_data);
            if (row >= float_col.size()) {
                float_col.resize(row + 1);
            }
            try {
                float_col[row] = std::stod(str);
            } catch (...) {
                float_col[row] = 0.0; // Default value for invalid conversion
            }
            break;
        }
        case DataType::STRING: {
            auto& string_col = std::get<StringColumn>(col_data);
            if (row >= string_col.size()) {
                string_col.resize(row + 1);
            }
            string_col[row] = str;
            break;
        }
        case DataType::BOOLEAN: {
            auto& bool_col = std::get<BoolColumn>(col_data);
            if (row >= bool_col.size()) {
                bool_col.resize(row + 1);
            }
            std::string lower_str = str;
            std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), 
                          [](unsigned char c){ return std::tolower(c); });
            bool_col[row] = (lower_str == "true" || lower_str == "1");
            break;
        }
    }
}

} // namespace storage
} // namespace gpu_dbms