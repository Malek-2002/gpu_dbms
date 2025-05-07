#include "parser/sql_parser.hpp"
#include "SQLParser.h"
#include "sql/SelectStatement.h"
#include "parser/bison_parser.h"
#include <iostream>
#include <functional>

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

// Flatten AND conditions into a list
std::vector<std::shared_ptr<Expression>> flattenAndConditions(std::shared_ptr<Expression> expr) {
    std::vector<std::shared_ptr<Expression>> conditions;
    if (!expr) {
        return conditions;
    }

    if (auto logical_expr = std::dynamic_pointer_cast<LogicalExpression>(expr)) {
        if (logical_expr->op == LogicalOperatorType::AND) {
            auto left_conditions = flattenAndConditions(logical_expr->left);
            auto right_conditions = flattenAndConditions(logical_expr->right);
            conditions.insert(conditions.end(), left_conditions.begin(), left_conditions.end());
            conditions.insert(conditions.end(), right_conditions.begin(), right_conditions.end());
        } else {
            conditions.push_back(expr);
        }
    } else {
        conditions.push_back(expr);
    }
    return conditions;
}

// Get TableRef objects from an expression
std::vector<TableRef> getTableRefsFromExpression(std::shared_ptr<Expression> expr, 
                                                 const std::vector<TableRef>& tables, 
                                                 const std::unordered_map<std::string, std::string>& alias_to_table_map) {
    std::set<std::string> table_identifiers;
    std::vector<TableRef> result;

    // Define recursive lambda to collect table identifiers
    std::function<void(std::shared_ptr<Expression>)> collectIdentifiers = [&](std::shared_ptr<Expression> e) {
        if (!e) return;
        if (auto col_expr = std::dynamic_pointer_cast<ColumnExpression>(e)) {
            if (!col_expr->column.alias.empty()) {
                table_identifiers.insert(col_expr->column.alias);
            } else if (!col_expr->column.table_name.empty()) {
                table_identifiers.insert(col_expr->column.table_name);
            }
        } else if (auto bin_expr = std::dynamic_pointer_cast<BinaryExpression>(e)) {
            collectIdentifiers(bin_expr->left);
            collectIdentifiers(bin_expr->right);
        } else if (auto logical_expr = std::dynamic_pointer_cast<LogicalExpression>(e)) {
            collectIdentifiers(logical_expr->left);
            collectIdentifiers(logical_expr->right);
        }
    };

    collectIdentifiers(expr);

    // Map identifiers to TableRef objects
    for (const auto& identifier : table_identifiers) {
        for (const auto& table : tables) {
            if (table.alias == identifier || table.table_name == identifier) {
                result.push_back(table);
            }
        }
    }

    return result;
}

void SQLParserWrapper::convertToQueryModel(const hsql::SQLParserResult& result) {
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
    
    printModel(0);
}

std::shared_ptr<QueryModel> convertSelectStatement(const hsql::SelectStatement* stmt) {
    auto model = std::make_shared<QueryModel>();
    
    std::unordered_map<std::string, std::string> alias_to_table_map;
    
    if (stmt->fromTable) {
        if (stmt->fromTable->type == hsql::kTableName) {
            TableRef table;
            table.table_name = stmt->fromTable->name;
            table.alias = stmt->fromTable->alias ? stmt->fromTable->alias->name : "";
            model->tables.push_back(table);
            
            if (!table.alias.empty()) {
                alias_to_table_map[table.alias] = table.table_name;
            }
            alias_to_table_map[table.table_name] = table.table_name;
        } else if (stmt->fromTable->type == hsql::kTableJoin) {
            auto join = stmt->fromTable->join;
            if (join && join->left && join->right) {
                if (join->left->type == hsql::kTableName) {
                    TableRef left_table;
                    left_table.table_name = join->left->getName();
                    left_table.alias = join->left->alias ? join->left->alias->name : "";
                    model->tables.push_back(left_table);
                    
                    if (!left_table.alias.empty()) {
                        alias_to_table_map[left_table.alias] = left_table.table_name;
                    }
                    alias_to_table_map[left_table.table_name] = left_table.table_name;
                }
                
                if (join->right->type == hsql::kTableName) {
                    TableRef right_table;
                    right_table.table_name = join->right->getName();
                    right_table.alias = join->right->alias ? join->right->alias->name : "";
                    model->tables.push_back(right_table);
                    
                    if (!right_table.alias.empty()) {
                        alias_to_table_map[right_table.alias] = right_table.table_name;
                    }
                    alias_to_table_map[right_table.table_name] = right_table.table_name;
                }
            }
        } else if (stmt->fromTable->type == hsql::kTableCrossProduct) {
            auto table_list = stmt->fromTable->list;
            if (table_list) {
                for (auto table_ref : *table_list) {
                    if (table_ref->type == hsql::kTableName) {
                        TableRef table;
                        table.table_name = table_ref->name;
                        table.alias = table_ref->alias ? table_ref->alias->name : "";
                        model->tables.push_back(table);
                        
                        if (!table.alias.empty()) {
                            alias_to_table_map[table.alias] = table.table_name;
                        }
                        alias_to_table_map[table.table_name] = table.table_name;
                    }
                }
            }
        } else if (stmt->fromTable->type == hsql::kTableSelect) {
            if (stmt->fromTable->select) {
                model->subquery = convertSelectStatement(stmt->fromTable->select);
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
    
    for (hsql::Expr* expr : *stmt->selectList) {
        model->select_list.push_back(convertExpression(expr, alias_to_table_map));
    }
    
    if (stmt->whereClause) {
        model->where_clause = convertExpression(stmt->whereClause, alias_to_table_map);
        if (!model->tables.empty()) {
            model->conditions = flattenAndConditions(model->where_clause);
            for (size_t i = 0; i < model->conditions.size(); ++i) {
                const auto& cond = model->conditions[i];
                auto table_refs = getTableRefsFromExpression(cond, model->tables, alias_to_table_map);
                if (table_refs.size() > 1) {
                    // Join condition
                    model->join_conditions.emplace_back(table_refs, i);
                }
                // Associate condition index with each referenced table
                for (const auto& table : table_refs) {
                    std::string key = table.alias.empty() ? table.table_name : table.alias;
                    model->table_specific_conditions[key].push_back(i);
                }
            }
        }
    }
    
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
            if (expr->opType == hsql::kOpAnd || expr->opType == hsql::kOpOr) {
                auto logical_expr = std::make_shared<LogicalExpression>();
                
                switch (expr->opType) {
                    case hsql::kOpAnd:
                        logical_expr->op = LogicalOperatorType::AND;
                        break;
                    case hsql::kOpOr:
                        logical_expr->op = LogicalOperatorType::OR;
                        break;
                    default:
                        return nullptr;
                }
                
                logical_expr->left = convertExpression(expr->expr, alias_to_table_map);
                logical_expr->right = convertExpression(expr->expr2, alias_to_table_map);
                return logical_expr;
            }
            
            auto bin_expr = std::make_shared<BinaryExpression>();
            
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
                    return nullptr;
            }
            
            bin_expr->left = convertExpression(expr->expr, alias_to_table_map);
            bin_expr->right = convertExpression(expr->expr2, alias_to_table_map);
            return bin_expr;
        }
        
        case hsql::kExprFunctionRef: {
            std::string func_name = expr->name;
            for (char& c : func_name) c = tolower(c);
            
            if (func_name == "count" || func_name == "sum" || func_name == "avg" ||
                func_name == "min" || func_name == "max") {
                auto agg_expr = std::make_shared<AggregateExpression>();
                
                if (func_name == "count") agg_expr->type = AggregateType::COUNT;
                else if (func_name == "sum") agg_expr->type = AggregateType::SUM;
                else if (func_name == "avg") agg_expr->type = AggregateType::AVG;
                else if (func_name == "min") agg_expr->type = AggregateType::MIN;
                else if (func_name == "max") agg_expr->type = AggregateType::MAX;
                
                if (func_name == "count" && (!expr->exprList || expr->exprList->empty())) {
                    // COUNT(*)
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

static void printIndent(int indent_level) {
    std::cout << std::string(indent_level * 2, ' ');
}

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

    if (!query_model_->select_list.empty()) {
        printIndent(indent_level + 1);
        std::cout << "Select List:" << std::endl;
        for (const auto& expr : query_model_->select_list) {
            printIndent(indent_level + 2);
            std::cout << "Expression:" << std::endl;
            printExpression(expr, indent_level + 3);
        }
    }

    if (!query_model_->conditions.empty()) {
        printIndent(indent_level + 1);
        std::cout << "Conditions:" << std::endl;
        for (size_t i = 0; i < query_model_->conditions.size(); ++i) {
            printIndent(indent_level + 2);
            std::cout << "Condition [" << i << "]:" << std::endl;
            printExpression(query_model_->conditions[i], indent_level + 3);
        }
    }

    if (!query_model_->join_conditions.empty()) {
        printIndent(indent_level + 1);
        std::cout << "Join Conditions:" << std::endl;
        for (const auto& [table_refs, cond_index] : query_model_->join_conditions) {
            printIndent(indent_level + 2);
            std::cout << "Condition [" << cond_index << "] between tables: ";
            for (const auto& table : table_refs) {
                std::cout << table.table_name;
                if (!table.alias.empty()) {
                    std::cout << " (" << table.alias << ")";
                }
                std::cout << " ";
            }
            std::cout << std::endl;
        }
    }

    if (!query_model_->table_specific_conditions.empty()) {
        printIndent(indent_level + 1);
        std::cout << "Table-Specific Conditions:" << std::endl;
        for (const auto& [table_key, cond_indices] : query_model_->table_specific_conditions) {
            printIndent(indent_level + 2);
            std::cout << "Table: " << table_key << std::endl;
            for (const auto& cond_index : cond_indices) {
                printIndent(indent_level + 3);
                std::cout << "Condition [" << cond_index << "]" << std::endl;
            }
        }
    }

    if (query_model_->where_clause) {
        printIndent(indent_level + 1);
        std::cout << "Where Clause:" << std::endl;
        printExpression(query_model_->where_clause, indent_level + 2);
    }

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