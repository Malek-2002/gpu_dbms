#ifndef GPU_DBMS_CLI_COMMAND_LINE_HPP
#define GPU_DBMS_CLI_COMMAND_LINE_HPP

#include <functional>
#include <vector>
#include <string>
#include <utility>
#include "execution/query_executer.hpp"
#include "storage/catalog.hpp"

namespace gpu_dbms {
namespace cli {

class CommandLine {
public:
    CommandLine();
    ~CommandLine();

    // Main execution loop
    void run();
    void run_e2e(int argc, char** argv);

    // Add custom command handler
    void addCommandHandler(const std::string& prefix, 
                          std::function<bool(const std::string&)> handler);

private:
    // Command handlers (prefix, function)
    std::vector<std::pair<std::string, std::function<bool(const std::string&)>>> command_handlers_;
    
    // Query executor
    execution::QueryExecutor executor_;

    // Display methods
    void displayWelcome();
    void displayHelp();

    // Command processing
    bool processCommand(const std::string& command);
    bool processSQLQuery(const std::string& query, std::string file_name = "");

    // Built-in command handlers
    bool processLoadCommand(const std::string& command);
    bool processTablesCommand(const std::string& command);
    bool processSchemaCommand(const std::string& command);
    bool processExitCommand(const std::string& command);
};

} // namespace cli
} // namespace gpu_dbms

#endif // GPU_DBMS_CLI_COMMAND_LINE_HPP