#include "cli/command_line.hpp"
#include "parser/sql_parser.hpp"
#include "execution/query_executer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace gpu_dbms {
namespace cli {

CommandLine::CommandLine() : executor_(storage::Catalog::getInstance()) {
    // Register built-in command handlers
    addCommandHandler(".load", [this](const std::string& cmd) { return processLoadCommand(cmd); });
    addCommandHandler(".tables", [this](const std::string& cmd) { return processTablesCommand(cmd); });
    addCommandHandler(".schema", [this](const std::string& cmd) { return processSchemaCommand(cmd); });
    addCommandHandler(".exit", [this](const std::string& cmd) { return processExitCommand(cmd); });
    addCommandHandler(".quit", [this](const std::string& cmd) { return processExitCommand(cmd); });
    addCommandHandler(".help", [this](const std::string& cmd) { displayHelp(); return true; });
}

CommandLine::~CommandLine() = default;

void CommandLine::run() {
    displayWelcome();
    
    std::string line;
    std::string query;
    bool in_multiline = false;
    
    while (true) {
        // Display prompt
        std::cout << (in_multiline ? "... " : "gpu-dbms> ");
        
        // Get line from user
        if (!std::getline(std::cin, line)) {
            break;
        }
        
        // Check if we're continuing a multi-line query
        if (in_multiline) {
            if (line == ";") {
                // End of query
                in_multiline = false;
                if (processSQLQuery(query)) {
                    query.clear();
                }
            } else {
                // Add to query
                query += " " + line;
                if (line.back() == ';') {
                    in_multiline = false;
                    if (processSQLQuery(query)) {
                        query.clear();
                    }
                }
            }
        } else {
            // New command or query
            if (line.empty()) {
                continue;
            }
            
            if (line.front() == '.') {
                // Command
                processCommand(line);
            } else {
                // SQL query
                query = line;
                if (line.back() != ';') {
                    in_multiline = true;
                } else {
                    processSQLQuery(query);
                    query.clear();
                }
            }
        }
    }
}

bool CommandLine::processCommand(const std::string& command) {
    for (const auto& handler : command_handlers_) {
        if (command.find(handler.first) == 0) {
            return handler.second(command);
        }
    }
    
    std::cout << "Unknown command: " << command << std::endl;
    return false;
}

void CommandLine::addCommandHandler(const std::string& prefix, 
                                   std::function<bool(const std::string&)> handler) {
    command_handlers_.emplace_back(prefix, handler);
}

void CommandLine::displayWelcome() {
    std::cout << "GPU-Accelerated DBMS v0.1" << std::endl;
    std::cout << "Enter \".help\" for usage hints." << std::endl;
}

void CommandLine::displayHelp() {
    std::cout << "Commands:" << std::endl;
    std::cout << "  .load FILENAME TABLE     Load CSV file into table" << std::endl;
    std::cout << "  .tables                  Show all tables" << std::endl;
    std::cout << "  .schema TABLE            Show schema for table" << std::endl;
    std::cout << "  .exit                    Exit this program" << std::endl;
    std::cout << "  .quit                    Exit this program" << std::endl;
    std::cout << "  .help                    Show this message" << std::endl;
    std::cout << std::endl;
    std::cout << "For SQL queries, end with a semicolon (;)" << std::endl;
    std::cout << "Use multiple lines by omitting the semicolon until the last line" << std::endl;
}

bool CommandLine::processLoadCommand(const std::string& command) {
    std::istringstream iss(command);
    std::string cmd, filename, table_name;
    
    iss >> cmd >> filename >> table_name;
    
    if (filename.empty() || table_name.empty()) {
        std::cout << "Usage: .load FILENAME TABLE" << std::endl;
        return false;
    }
    
    try {
        storage::Catalog::getInstance().loadFromCSV(table_name, filename);
        std::cout << "Loaded data from " << filename << " into table " << table_name << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return false;
    }
}

bool CommandLine::processTablesCommand(const std::string& command) {
    try {
        auto table_names = storage::Catalog::getInstance().getAllTableNames();
        
        if (table_names.empty()) {
            std::cout << "No tables found" << std::endl;
        } else {
            for (const auto& name : table_names) {
                std::cout << name << std::endl;
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return false;
    }
}

bool CommandLine::processSchemaCommand(const std::string& command) {
    std::istringstream iss(command);
    std::string cmd, table_name;
    
    iss >> cmd >> table_name;
    
    if (table_name.empty()) {
        std::cout << "Usage: .schema TABLE" << std::endl;
        return false;
    }
    
    try {
        auto schema = storage::Catalog::getInstance().getSchema(table_name);
        const auto& columns = schema->getColumns();
        
        std::cout << "Schema for table " << table_name << ":" << std::endl;
        std::cout << std::left << std::setw(20) << "Column Name" 
                  << std::setw(10) << "Type" 
                  << std::setw(5) << "PK" 
                  << std::setw(5) << "FK" 
                  << "Referenced Table.Column" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& col : columns) {
            std::string type_str;
            switch (col.type) {
                case storage::DataType::INT: type_str = "INT"; break;
                case storage::DataType::FLOAT: type_str = "FLOAT"; break;
                case storage::DataType::STRING: type_str = "STRING"; break;
                case storage::DataType::BOOLEAN: type_str = "BOOL"; break;
            }
            
            std::string ref_str;
            if (col.is_foreign_key) {
                ref_str = col.referenced_table + "." + col.referenced_column;
            }
            
            std::cout << std::left << std::setw(20) << col.name 
                      << std::setw(10) << type_str 
                      << std::setw(5) << (col.is_primary_key ? "Yes" : "No")
                      << std::setw(5) << (col.is_foreign_key ? "Yes" : "No")
                      << ref_str << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return false;
    }
}

bool CommandLine::processExitCommand(const std::string& command) {
    std::cout << "Goodbye!" << std::endl;
    exit(0);
    return true;
}

bool CommandLine::processSQLQuery(const std::string& query) {
    // Remove trailing semicolon if present
    std::string cleaned_query = query;
    if (!cleaned_query.empty() && cleaned_query.back() == ';') {
        cleaned_query.pop_back();
    }
    
    // Trim whitespace
    cleaned_query.erase(0, cleaned_query.find_first_not_of(" \t\n\r"));
    cleaned_query.erase(cleaned_query.find_last_not_of(" \t\n\r") + 1);
    
    if (cleaned_query.empty()) {
        return true;
    }
    
    try {
        // Parse SQL query
        parser::SQLParserWrapper parser;
        if (!parser.parse(cleaned_query)) {
            std::cout << "Error: " << parser.getErrorMsg() << std::endl;
            return false;
        }
        
        // Execute the query
        auto query_model = parser.getQueryModel();
        auto result = executor_.execute(query_model);
        
        // Display the result
        if (!result || !result->getData()) {
            std::cout << "No results returned" << std::endl;
            return true;
        }
        
        auto schema = result->getSchema();
        auto table = result->getData();
        size_t num_rows = result->numRows();
        
        // Print column headers
        const auto& columns = schema->getColumns();
        std::vector<size_t> col_widths;
        for (const auto& col : columns) {
            size_t width = std::max(col.name.length(), size_t(10)); // Minimum width
            col_widths.push_back(width);
            std::cout << std::left << std::setw(width) << col.name << " |";
        }
        std::cout << std::endl;
        
        // Print separator
        for (size_t width : col_widths) {
            std::cout << std::string(width, '-') << "-+";
        }
        std::cout << std::endl;
        
        // Print rows
        for (size_t row = 0; row < num_rows; ++row) {
            for (size_t col_idx = 0; col_idx < columns.size(); ++col_idx) {
                const auto& col_info = columns[col_idx];
                const auto& col_name = col_info.name;
                const auto& col_data = table->getColumn(col_name);
                
                std::string value;
                switch (col_info.type) {
                    case storage::DataType::INT: {
                        auto data = std::get<storage::IntColumn>(col_data);
                        value = std::to_string(data[row]);
                        break;
                    }
                    case storage::DataType::FLOAT: {
                        auto data = std::get<storage::FloatColumn>(col_data);
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(2) << data[row];
                        value = ss.str();
                        break;
                    }
                    case storage::DataType::STRING: {
                        auto data = std::get<storage::StringColumn>(col_data);
                        value = data[row].empty() ? "NULL" : data[row];
                        break;
                    }
                    case storage::DataType::BOOLEAN: {
                        auto data = std::get<storage::BoolColumn>(col_data);
                        value = data[row] ? "TRUE" : "FALSE";
                        break;
                    }
                    default:
                        value = "UNKNOWN";
                }
                
                std::cout << std::left << std::setw(col_widths[col_idx]) << value << " |";
            }
            std::cout << std::endl;
        }
        
        std::cout << num_rows << " row" << (num_rows == 1 ? "" : "s") << " returned" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace cli
} // namespace gpu_dbms