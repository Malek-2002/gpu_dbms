#include "parser/sql_parser.hpp"
#include "SQLParser.h"
#include "sql/SelectStatement.h"
#include "parser/bison_parser.h"
#include <iostream>

namespace gpu_dbms {
namespace parser {

SQLParserWrapper::SQLParserWrapper() : query_model_(nullptr) {}

SQLParserWrapper::~SQLParserWrapper() = default;

bool SQLParserWrapper::parse(const std::string& query) {
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(query, &result);
    
    if (!result.isValid()) {
        error_msg_ = result.errorMsg();
        return false;
    }
    
    convertToQueryModel(result);
    return true;
}

std::shared_ptr<QueryModel> SQLParserWrapper::getQueryModel() const {
    return query_model_;
}

std::string SQLParserWrapper::getErrorMsg() const {
    return error_msg_;
}

// Helper functions for convertToQueryModel
std::shared_ptr<Expression> convertExpression(const hsql::Expr* expr, const std::unordered_map<std::string, std::string>& alias_to_table_map);
std::shared_ptr<QueryModel> convertSelectStatement(const hsql::SelectStatement* stmt);

void SQLParserWrapper::convertToQueryModel(const hsql::SQLParserResult& result) {
    // We only handle SELECT statements in this phase
    if (result.size() != 1) {
        error_msg_ = "Only single queries are supported";
        return;
    }
    
    const hsql::SQLStatement* stmt = result.getStatement(0);
    if (stmt->type() != hsql::kStmtSelect) {
        error_msg_ = "Only SELECT statements are supported";
        return;
    }
    
    const hsql::SelectStatement* select_stmt = static_cast<const hsql::SelectStatement*>(stmt);
    query_model_ = convertSelectStatement(select_stmt);

    // printModel(0);
}

std::shared_ptr<QueryModel> convertSelectStatement(const hsql::SelectStatement* stmt) {
    auto model = std::make_shared<QueryModel>();
    
    // Create a mapping from aliases to actual table names
    std::unordered_map<std::string, std::string> alias_to_table_map;
    
    // Convert FROM clause (tables)
    if (stmt->fromTable) {
        if (stmt->fromTable->type == hsql::kTableName) {
            TableRef table;
            table.table_name = stmt->fromTable->name;
            table.alias = stmt->fromTable->alias ? stmt->fromTable->alias->name : "";
            model->tables.push_back(table);
            
            // Store the mapping
            if (!table.alias.empty()) {
                alias_to_table_map[table.alias] = table.table_name;
            }
            alias_to_table_map[table.table_name] = table.table_name;
        } else if (stmt->fromTable->type == hsql::kTableJoin) {
            // Handle explicit joins
            auto join = stmt->fromTable->join;
            if (join && join->left && join->right) {
                // Process left table
                if (join->left->type == hsql::kTableName) {
                    TableRef left_table;
                    left_table.table_name = join->left->getName();
                    left_table.alias = join->left->alias ? join->left->alias->name : "";
                    model->tables.push_back(left_table);
                    
                    // Store the mapping
                    if (!left_table.alias.empty()) {
                        alias_to_table_map[left_table.alias] = left_table.table_name;
                    }
                    alias_to_table_map[left_table.table_name] = left_table.table_name;
                }
                
                // Process right table
                if (join->right->type == hsql::kTableName) {
                    TableRef right_table;
                    right_table.table_name = join->right->getName();
                    right_table.alias = join->right->alias ? join->right->alias->name : "";
                    model->tables.push_back(right_table);
                    
                    // Store the mapping
                    if (!right_table.alias.empty()) {
                        alias_to_table_map[right_table.alias] = right_table.table_name;
                    }
                    alias_to_table_map[right_table.table_name] = right_table.table_name;
                }
            }
        } else if (stmt->fromTable->type == hsql::kTableCrossProduct) {
            // Handle cross product (comma-separated tables)
            auto table_list = stmt->fromTable->list;
            if (table_list) {
                for (auto table_ref : *table_list) {
                    if (table_ref->type == hsql::kTableName) {
                        TableRef table;
                        table.table_name = table_ref->name;
                        table.alias = table_ref->alias ? table_ref->alias->name : "";
                        model->tables.push_back(table);
                        
                        // Store the mapping
                        if (!table.alias.empty()) {
                            alias_to_table_map[table.alias] = table.table_name;
                        }
                        alias_to_table_map[table.table_name] = table.table_name;
                    }
                }
            }
        } else if (stmt->fromTable->type == hsql::kTableSelect) {
            // Handle subquery in FROM clause
            if (stmt->fromTable->select) {
                model->subquery = convertSelectStatement(stmt->fromTable->select);
                // Add subquery alias to tables and map
                if (stmt->fromTable->alias) {
                    TableRef table;
                    table.table_name = "";
                    table.alias = stmt->fromTable->alias->name;
                    model->tables.push_back(table);
                    alias_to_table_map[table.alias] = table.alias;
                }
            }
        }
    }
    
    // Convert SELECT list
    for (hsql::Expr* expr : *stmt->selectList) {
        model->select_list.push_back(convertExpression(expr, alias_to_table_map));
    }
    
    // Convert WHERE clause
    if (stmt->whereClause) {
        model->where_clause = convertExpression(stmt->whereClause, alias_to_table_map);
    }
    
    // Convert ORDER BY
    if (stmt->order) {
        for (hsql::OrderDescription* order : *stmt->order) {
            SortOrder sort_order = order->type == hsql::kOrderAsc ? SortOrder::ASC : SortOrder::DESC;
            model->order_by.emplace_back(convertExpression(order->expr, alias_to_table_map), sort_order);
        }
    }
    
    return model;
}

std::shared_ptr<Expression> convertExpression(const hsql::Expr* expr, 
                                              const std::unordered_map<std::string, std::string>& alias_to_table_map) {
    if (!expr) return nullptr;
    
    switch (expr->type) {
        case hsql::kExprColumnRef: {
            auto col_expr = std::make_shared<ColumnExpression>();
            col_expr->column.column_name = expr->name;
            if (expr->table) {
                std::string table_alias = expr->table;
                // Look up the actual table name from the alias
                auto it = alias_to_table_map.find(table_alias);
                if (it != alias_to_table_map.end()) {
                    col_expr->column.table_name = it->second;
                    col_expr->column.alias = table_alias;
                } else {
                    col_expr->column.table_name = table_alias;
                    col_expr->column.alias = "";
                }
            } else {
                col_expr->column.table_name = "";
                col_expr->column.alias = "";
            }
            return col_expr;
        }
        
        case hsql::kExprLiteralInt: {
            auto const_expr = std::make_shared<ConstantExpression>();
            const_expr->type = ConstantExpression::ValueType::INTEGER;
            const_expr->int_value = expr->ival;
            return const_expr;
        }
        
        case hsql::kExprLiteralFloat: {
            auto const_expr = std::make_shared<ConstantExpression>();
            const_expr->type = ConstantExpression::ValueType::FLOAT;
            const_expr->float_value = expr->fval;
            return const_expr;
        }
        
        case hsql::kExprLiteralString: {
            auto const_expr = std::make_shared<ConstantExpression>();
            const_expr->type = ConstantExpression::ValueType::STRING;
            const_expr->string_value = expr->name;
            return const_expr;
        }
        
        case hsql::kExprOperator: {
            // Handle logical operators (AND, OR)
            if (expr->opType == hsql::kOpAnd || expr->opType == hsql::kOpOr) {
                auto logical_expr = std::make_shared<LogicalExpression>();
                
                // Convert logical operator type
                switch (expr->opType) {
                    case hsql::kOpAnd:
                        logical_expr->op = LogicalOperatorType::AND;
                        break;
                    case hsql::kOpOr:
                        logical_expr->op = LogicalOperatorType::OR;
                        break;
                    default:
                        return nullptr; // Should not happen
                }
                
                logical_expr->left = convertExpression(expr->expr, alias_to_table_map);
                logical_expr->right = convertExpression(expr->expr2, alias_to_table_map);
                return logical_expr;
            }
            
            // Handle comparison operators
            auto bin_expr = std::make_shared<BinaryExpression>();
            
            // Convert comparison operator type
            switch (expr->opType) {
                case hsql::kOpEquals:
                    bin_expr->op = OperatorType::EQUALS;
                    break;
                case hsql::kOpNotEquals:
                    bin_expr->op = OperatorType::NOT_EQUALS;
                    break;
                case hsql::kOpLess:
                    bin_expr->op = OperatorType::LESS_THAN;
                    break;
                case hsql::kOpGreater:
                    bin_expr->op = OperatorType::GREATER_THAN;
                    break;
                case hsql::kOpLessEq:
                    bin_expr->op = OperatorType::LESS_EQUALS;
                    break;
                case hsql::kOpGreaterEq:
                    bin_expr->op = OperatorType::GREATER_EQUALS;
                    break;
                default:
                    return nullptr; // Unsupported operator
            }
            
            bin_expr->left = convertExpression(expr->expr, alias_to_table_map);
            bin_expr->right = convertExpression(expr->expr2, alias_to_table_map);
            return bin_expr;
        }
        
        case hsql::kExprFunctionRef: {
            // Handle aggregate functions
            std::string func_name = expr->name;
            for (char& c : func_name) c = tolower(c); // Convert to lowercase
            
            if (func_name == "count" || func_name == "sum" || func_name == "avg" ||
                func_name == "min" || func_name == "max") {
                auto agg_expr = std::make_shared<AggregateExpression>();
                
                if (func_name == "count") agg_expr->type = AggregateType::COUNT;
                else if (func_name == "sum") agg_expr->type = AggregateType::SUM;
                else if (func_name == "avg") agg_expr->type = AggregateType::AVG;
                else if (func_name == "min") agg_expr->type = AggregateType::MIN;
                else if (func_name == "max") agg_expr->type = AggregateType::MAX;
                
                // Handle COUNT(*)
                if (func_name == "count" && (!expr->exprList || expr->exprList->empty())) {
                    // COUNT(*) has no expression
                } else if (expr->exprList && expr->exprList->size() > 0) {
                    agg_expr->expr = convertExpression(expr->exprList->at(0), alias_to_table_map);
                }
                
                if (expr->alias) {
                    agg_expr->alias = expr->alias;
                }
                
                return agg_expr;
            }
            break;
        }
        
        default:
            break;
    }
    
    return nullptr;
}

// Helper function to print indentation
static void printIndent(int indent_level) {
    std::cout << std::string(indent_level * 2, ' ');
}

// Helper function to convert OperatorType to string
static std::string operatorToString(OperatorType op) {
    switch (op) {
        case OperatorType::EQUALS: return "=";
        case OperatorType::NOT_EQUALS: return "!=";
        case OperatorType::LESS_THAN: return "<";
        case OperatorType::GREATER_THAN: return ">";
        case OperatorType::LESS_EQUALS: return "<=";
        case OperatorType::GREATER_EQUALS: return ">=";
        default: return "UNKNOWN";
    }
}

// Helper function to convert AggregateType to string
static std::string aggregateToString(AggregateType type) {
    switch (type) {
        case AggregateType::COUNT: return "COUNT";
        case AggregateType::SUM: return "SUM";
        case AggregateType::AVG: return "AVG";
        case AggregateType::MIN: return "MIN";
        case AggregateType::MAX: return "MAX";
        default: return "UNKNOWN";
    }
}

// Helper function to convert SortOrder to string
static std::string sortOrderToString(SortOrder order) {
    return order == SortOrder::ASC ? "ASC" : "DESC";
}

static void printExpression(const std::shared_ptr<Expression>& expr, int indent_level) {
    if (!expr) {
        printIndent(indent_level);
        std::cout << "Null Expression" << std::endl;
        return;
    }

    if (auto col_expr = std::dynamic_pointer_cast<ColumnExpression>(expr)) {
        printIndent(indent_level);
        std::cout << "ColumnExpression:" << std::endl;
        printIndent(indent_level + 1);
        std::cout << "Column: " << col_expr->column.column_name;
        if (!col_expr->column.table_name.empty()) {
            std::cout << " (Table: " << col_expr->column.table_name;
            if (!col_expr->column.alias.empty() && col_expr->column.alias != col_expr->column.table_name) {
                std::cout << ", Alias: " << col_expr->column.alias;
            }
            std::cout << ")";
        }
        std::cout << std::endl;
    }
    else if (auto const_expr = std::dynamic_pointer_cast<ConstantExpression>(expr)) {
        printIndent(indent_level);
        std::cout << "ConstantExpression:" << std::endl;
        printIndent(indent_level + 1);
        std::cout << "Type: ";
        switch (const_expr->type) {
            case ConstantExpression::ValueType::INTEGER:
                std::cout << "INTEGER, Value: " << const_expr->int_value;
                break;
            case ConstantExpression::ValueType::FLOAT:
                std::cout << "FLOAT, Value: " << const_expr->float_value;
                break;
            case ConstantExpression::ValueType::STRING:
                std::cout << "STRING, Value: \"" << const_expr->string_value << "\"";
                break;
            default:
                std::cout << "UNKNOWN";
        }
        std::cout << std::endl;
    }
    else if (auto bin_expr = std::dynamic_pointer_cast<BinaryExpression>(expr)) {
        printIndent(indent_level);
        std::cout << "BinaryExpression:" << std::endl;
        printIndent(indent_level + 1);
        std::cout << "Operator: " << operatorToString(bin_expr->op) << std::endl;
        printIndent(indent_level + 1);
        std::cout << "Left:" << std::endl;
        printExpression(bin_expr->left, indent_level + 2);
        printIndent(indent_level + 1);
        std::cout << "Right:" << std::endl;
        printExpression(bin_expr->right, indent_level + 2);
    }
    else if (auto logical_expr = std::dynamic_pointer_cast<LogicalExpression>(expr)) {
        printIndent(indent_level);
        std::cout << "LogicalExpression:" << std::endl;
        printIndent(indent_level + 1);
        std::cout << "Operator: " << (logical_expr->op == LogicalOperatorType::AND ? "AND" : "OR") << std::endl;
        printIndent(indent_level + 1);
        std::cout << "Left:" << std::endl;
        printExpression(logical_expr->left, indent_level + 2);
        printIndent(indent_level + 1);
        std::cout << "Right:" << std::endl;
        printExpression(logical_expr->right, indent_level + 2);
    }
    else if (auto agg_expr = std::dynamic_pointer_cast<AggregateExpression>(expr)) {
        printIndent(indent_level);
        std::cout << "AggregateExpression:" << std::endl;
        printIndent(indent_level + 1);
        std::cout << "Type: " << aggregateToString(agg_expr->type) << std::endl;
        if (!agg_expr->alias.empty()) {
            printIndent(indent_level + 1);
            std::cout << "Alias: " << agg_expr->alias << std::endl;
        }
        if (agg_expr->expr) {
            printIndent(indent_level + 1);
            std::cout << "Expression:" << std::endl;
            printExpression(agg_expr->expr, indent_level + 2);
        }
    }
    else {
        printIndent(indent_level);
        std::cout << "Unknown Expression Type" << std::endl;
    }
}

void SQLParserWrapper::printModel(int indent_level) const {
    if (!query_model_) {
        printIndent(indent_level);
        std::cout << "No query model available" << std::endl;
        return;
    }

    printIndent(indent_level);
    std::cout << "QueryModel:" << std::endl;

    // Print tables
    if (!query_model_->tables.empty()) {
        printIndent(indent_level + 1);
        std::cout << "Tables:" << std::endl;
        for (const auto& table : query_model_->tables) {
            printIndent(indent_level + 2);
            std::cout << "Table: " << table.table_name;
            if (!table.alias.empty()) {
                std::cout << " (Alias: " << table.alias << ")";
            }
            std::cout << std::endl;
        }
    }

    // Print select list
    if (!query_model_->select_list.empty()) {
        printIndent(indent_level + 1);
        std::cout << "Select List:" << std::endl;
        for (const auto& expr : query_model_->select_list) {
            printIndent(indent_level + 2);
            std::cout << "Expression:" << std::endl;
            printExpression(expr, indent_level + 3);
        }
    }

    // Print where clause
    if (query_model_->where_clause) {
        printIndent(indent_level + 1);
        std::cout << "Where Clause:" << std::endl;
        printExpression(query_model_->where_clause, indent_level + 2);
    }

    // Print order by
    if (!query_model_->order_by.empty()) {
        printIndent(indent_level + 1);
        std::cout << "Order By:" << std::endl;
        for (const auto& order : query_model_->order_by) {
            printIndent(indent_level + 2);
            std::cout << "Sort:" << std::endl;
            printIndent(indent_level + 3);
            std::cout << "Expression:" << std::endl;
            printExpression(order.first, indent_level + 4);
            printIndent(indent_level + 3);
            std::cout << "Order: " << sortOrderToString(order.second) << std::endl;
        }
    }

    // Print subquery
    if (query_model_->subquery) {
        printIndent(indent_level + 1);
        std::cout << "Subquery:" << std::endl;
        SQLParserWrapper sub_parser;
        sub_parser.query_model_ = query_model_->subquery;
        sub_parser.printModel(indent_level + 2);
    }
}

} // namespace parser
} // namespace gpu_dbms