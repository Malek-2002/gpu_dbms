#include "cli/command_line.hpp"
#include "parser/sql_parser.hpp"
#include "execution/query_executer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

// For e2e test
#include <filesystem>
namespace fs = std::filesystem;
#define TEAM_NAME "Team8"

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

void CommandLine::run_e2e(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_folder> <query_file>" << std::endl;
        return;
    }
    std::string input_folder = argv[1];
    std::string query_file_path = argv[2];

    std::cout << "Starting E2E test..." << std::endl;

    // Extract the base name from the query file path (drop directory and extension)
    std::string base_name = fs::path(query_file_path).stem().string();  // e.g., "q1" from "queries/q1.sql"

    // Load all CSV files in the input folder
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            const std::string path = entry.path().string();
            if (path.size() >= 4 && path.substr(path.size() - 4) == ".csv") {
                std::string load_command = ".load " + path;
                std::cout << "Executing: " << load_command << std::endl;
                processLoadCommand(load_command);
            }
        }
    }

    // Open and read the query file
    std::ifstream query_file(query_file_path);
    if (!query_file.is_open()) {
        std::cerr << "Error: Failed to open query file: " << query_file_path << std::endl;
        return;
    }

    std::cout << "Executing queries from: " << query_file_path << std::endl;
    std::string line, full_query;
    while (std::getline(query_file, line)) {
        // Trim line
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line.empty()) continue;

        full_query += line + " ";

        // Check for end of query
        if (!line.empty() && line.back() == ';') {
            std::string output_file = TEAM_NAME + std::string("_") + base_name + ".csv";
            processSQLQuery(full_query, output_file);
            full_query.clear();
        }
    }

    if (!full_query.empty()) {
        std::cout << "Warning: Query did not end with ';': " << full_query << std::endl;
    }

    std::cout << "E2E test completed." << std::endl;
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
    std::string cmd, filename;
    
    iss >> cmd >> filename;
    
    if (filename.empty()) {
        std::cout << "Usage: .load FILENAME" << std::endl;
        return false;
    }

    // Extract just the filename (no path)
    size_t slash_pos = filename.find_last_of("/\\");
    std::string table_name = (slash_pos != std::string::npos)
                             ? filename.substr(slash_pos + 1)
                             : filename;

    // Remove .csv extension
    size_t dot_pos = table_name.rfind(".csv");
    if (dot_pos != std::string::npos && dot_pos == table_name.length() - 4) {
        table_name = table_name.substr(0, dot_pos);
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

bool CommandLine::processSQLQuery(const std::string& query, std::string file_name) {
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

        if (!result || !result->getData()) {
            std::cout << "No results returned" << std::endl;
            return true;
        }

        auto schema = result->getSchema();
        auto table = result->getData();
        size_t num_rows = result->numRows();
        const auto& columns = schema->getColumns();

        std::vector<size_t> col_widths;
        for (const auto& col : columns) {
            size_t width = std::max(col.name.length(), size_t(10));
            col_widths.push_back(width);
            std::cout << std::left << std::setw(width) << col.name << " |";
        }
        std::cout << std::endl;

        for (size_t width : col_widths) {
            std::cout << std::string(width, '-') << "-+";
        }
        std::cout << std::endl;

        // Prepare CSV file if needed
        std::ofstream csv_file;
        if (!file_name.empty()) {
            csv_file.open(file_name);
            if (!csv_file.is_open()) {
                std::cerr << "Error: Could not open file " << file_name << " for writing." << std::endl;
                return false;
            }

            // Write CSV headers
            for (size_t i = 0; i < columns.size(); ++i) {
                csv_file << columns[i].name;
                if (i < columns.size() - 1) csv_file << ",";
            }
            csv_file << "\n";
        }

        // Print and/or write rows
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

                if (!file_name.empty()) {
                    // Escape value for CSV if needed (e.g., quotes, commas)
                    if (value.find(',') != std::string::npos || value.find('"') != std::string::npos) {
                        std::string escaped = "\"";
                        for (char c : value) {
                            if (c == '"') escaped += "\"\"";
                            else escaped += c;
                        }
                        escaped += "\"";
                        csv_file << escaped;
                    } else {
                        csv_file << value;
                    }

                    if (col_idx < columns.size() - 1) csv_file << ",";
                }
            }
            std::cout << std::endl;
            if (!file_name.empty()) {
                csv_file << "\n";
            }
        }

        std::cout << num_rows << " row" << (num_rows == 1 ? "" : "s") << " returned" << std::endl;

        if (!file_name.empty()) {
            csv_file.close();
            std::cout << "Results written to: " << file_name << std::endl;
        }

        return true;
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return false;
    }
}

} // namespace cli
} // namespace gpu_dbms