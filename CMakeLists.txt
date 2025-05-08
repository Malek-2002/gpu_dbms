cmake_minimum_required(VERSION 3.10)
project(gpu_dbms)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define source files for the main executable
set(SOURCE_FILES
    src/main.cpp
    src/cli/command_line.cpp
    src/parser/sql_parser.cpp
    src/storage/csv_parser.cpp
    src/storage/schema.cpp
    src/storage/table.cpp
    src/storage/catalog.cpp
    src/execution/expression_evaluator.cpp
    src/execution/query_executer.cpp 
    src/execution/query_plan.cpp
    src/execution/result.cpp
    src/execution/operators/select_operator.cpp
    src/execution/operators/filter_operator.cpp
    src/execution/operators/sort_operator.cpp
    src/execution/operators/aggregate_operator.cpp
    src/execution/operators/join_operator.cpp
)

# Create the main executable
add_executable(gpu_dbms ${SOURCE_FILES})

# Set output binary to the main project directory
set_target_properties(gpu_dbms PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Define include directories
set(INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/sql-parser/src
)

# Set include directories for the executable
target_include_directories(gpu_dbms PRIVATE ${INCLUDE_DIRS})

# Find the sqlparser library
find_library(SQLPARSER_LIB sqlparser PATHS ${CMAKE_SOURCE_DIR}/extern/sql-parser)
if(NOT SQLPARSER_LIB)
    message(FATAL_ERROR "sqlparser library not found in ${CMAKE_SOURCE_DIR}/extern/sql-parser")
endif()

# Find nlohmann_json
find_package(nlohmann_json QUIET)
if (nlohmann_json_FOUND)
    target_link_libraries(gpu_dbms PRIVATE nlohmann_json::nlohmann_json)
else()
    target_include_directories(gpu_dbms PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

# Link the executable against the sqlparser library
target_link_libraries(gpu_dbms PRIVATE ${SQLPARSER_LIB})