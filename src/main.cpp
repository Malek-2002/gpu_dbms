#include "cli/command_line.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        gpu_dbms::cli::CommandLine cli;

        // Check if --cli is one of the arguments
        bool cli_mode = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--cli") {
                cli_mode = true;
                break;
            }
        }

        if (cli_mode) {
            cli.run();
        } else {
            cli.run_e2e(argc, argv);
        }

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
