#pragma once

#include <memory>
#include <string>
#include <vector>

#include "SQLParser.h"
#include "SQLParserResult.h"
#include "parser/query_model.hpp"

namespace gpu_dbms {
namespace parser {

// Forward declarations for helper functions
std::vector<std::shared_ptr<Expression>> flattenAndConditions(std::shared_ptr<Expression> expr);
std::vector<TableRef> getTableRefsFromExpression(std::shared_ptr<Expression> expr, 
    const std::vector<TableRef>& tables, 
    const std::unordered_map<std::string, std::string>& alias_to_table_map);
std::shared_ptr<Expression> convertExpression(const hsql::Expr* expr, 
    const std::unordered_map<std::string, std::string>& alias_to_table_map,
    QueryModel* model);

class SQLParserWrapper {
public:
    SQLParserWrapper();
    ~SQLParserWrapper();
    
    bool parse(const std::string& query);
    std::shared_ptr<QueryModel> getQueryModel() const;
    std::string getErrorMsg() const;
    
    void printModel(int indent_level) const;
    
    std::shared_ptr<QueryModel> query_model_;
private:
    std::string error_msg_;
    
    void convertToQueryModel(const hsql::SQLParserResult& result);
};

} // namespace parser
} // namespace gpu_dbms