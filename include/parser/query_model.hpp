#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

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
    GREATER_EQUALS,
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

struct QueryModel {
    std::vector<TableRef> tables;
    std::vector<std::shared_ptr<Expression>> select_list;
    std::shared_ptr<Expression> where_clause;
    std::vector<std::pair<std::shared_ptr<Expression>, SortOrder>> order_by;
    std::shared_ptr<QueryModel> subquery;
    
    bool hasSubquery() const { return subquery != nullptr; }
};

} // namespace parser
} // namespace gpu_dbms