#include "cli/command_line.hpp"
#include <iostream>
#define CLI_FLAG false

int main(int argc, char** argv) {
    if (CLI_FLAG) {
        try {
            gpu_dbms::cli::CommandLine cli;
            cli.run();
        } catch (const std::exception& e) {
            std::cerr << "Fatal error: " << e.what() << std::endl;
            return 1;
        }
    } else {
        try {
            gpu_dbms::cli::CommandLine cli;
            cli.run_e2e(argc, argv);
        } catch (const std::exception& e) {
            std::cerr << "Fatal error: " << e.what() << std::endl;
            return 1;
        }
    }
    
    return 0;
}