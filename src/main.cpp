#include "cli/command_line.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        gpu_dbms::cli::CommandLine cli;
        cli.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}