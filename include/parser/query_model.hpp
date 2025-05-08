#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

namespace gpu_dbms {
namespace parser {

enum class OperationType {
    SELECT,
    PROJECT,
    FILTER,
    JOIN,
    AGGREGATE,
    SORT,
    SUBQUERY
};

enum class JoinType {
    INNER,
    LEFT,
    RIGHT,
    FULL
};

enum class AggregateType {
    COUNT,
    SUM,
    AVG,
    MIN,
    MAX
};

enum class SortOrder {
    ASC,
    DESC
};

enum class OperatorType {
    EQUALS,
    NOT_EQUALS,
    LESS_THAN,
    GREATER_THAN,
    LESS_EQUALS,
    GREATER_EQUALS
};

enum class LogicalOperatorType {
    AND,
    OR
};

struct TableRef {
    std::string table_name;
    std::string alias;
};

struct ColumnRef {
    std::string column_name;
    std::string table_name;  // Can be empty
    std::string alias;       // Can be empty
};

struct Expression {
    virtual ~Expression() = default;
};

struct ColumnExpression : public Expression {
    ColumnRef column;
    std::string alias;  // Can be empty
};

struct ConstantExpression : public Expression {
    enum class ValueType {
        INTEGER,
        FLOAT,
        STRING,
        BOOLEAN,
        NULL_VALUE
    };
    
    ValueType type;
    union {
        int64_t int_value;
        double float_value;
        bool bool_value;
    };
    std::string string_value;  // Used for string type
};

struct BinaryExpression : public Expression {
    OperatorType op;
    std::shared_ptr<Expression> left;
    std::shared_ptr<Expression> right;
};

struct LogicalExpression : public Expression {
    LogicalOperatorType op;
    std::shared_ptr<Expression> left;
    std::shared_ptr<Expression> right;
};

struct AggregateExpression : public Expression {
    AggregateType type;
    std::shared_ptr<Expression> expr;
    std::string alias;
};

struct SubqueryExpression : public Expression {
    size_t subquery_index;  // Index into QueryModel::subqueries
};

struct QueryModel {
    OperationType type = OperationType::SELECT;  // Default to SELECT
    std::vector<TableRef> tables;
    std::vector<std::shared_ptr<Expression>> select_list;
    std::shared_ptr<Expression> where_clause;
    std::vector<std::pair<std::shared_ptr<Expression>, SortOrder>> order_by;
    std::shared_ptr<QueryModel> subquery;  // For FROM clause subquery
    std::vector<std::shared_ptr<QueryModel>> subqueries;  // For subqueries in expressions
    // Store all conditions once
    std::vector<std::shared_ptr<Expression>> conditions;
    // Join conditions reference indices into conditions
    std::vector<std::pair<std::vector<TableRef>, size_t>> join_conditions;
    // Table-specific conditions reference indices into conditions
    std::unordered_map<std::string, std::vector<size_t>> table_specific_conditions;
    
    bool hasSubquery() const { return subquery != nullptr || !subqueries.empty(); }
};

} // namespace parser
} // namespace gpu_dbms 