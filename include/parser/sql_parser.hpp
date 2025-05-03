#pragma once

#include <memory>
#include <string>
#include <vector>

#include "SQLParser.h"
#include "SQLParserResult.h"
#include "parser/query_model.hpp"

namespace gpu_dbms {
namespace parser {

class SQLParserWrapper {
public:
    SQLParserWrapper();
    ~SQLParserWrapper();

    // Parse SQL query string
    bool parse(const std::string& query);
    
    // Get parsed query model
    std::shared_ptr<QueryModel> getQueryModel() const;
    
    // Get error message if parsing failed
    std::string getErrorMsg() const;

    // Print the AST of the parsed query
    void printModel(int indent_level = 0) const;

private:
    // Convert hsql::SQLParserResult to our QueryModel
    void convertToQueryModel(const hsql::SQLParserResult& result);
    
    std::shared_ptr<QueryModel> query_model_;
    std::string error_msg_;
};

} // namespace parser
} // namespace gpu_dbms