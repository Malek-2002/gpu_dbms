#pragma once

#include <memory>
#include <string>
#include <variant>
#include "parser/query_model.hpp"
#include "storage/table.hpp"
#include "storage/schema.hpp"

namespace gpu_dbms {
namespace execution {

// Variant to hold possible expression evaluation results
using Value = std::variant<int64_t, double, std::string, bool, std::monostate>;

class ExpressionEvaluator {
public:
    ExpressionEvaluator(std::shared_ptr<storage::Table> table);

    // Evaluate an expression for a specific row
    Value evaluate(const parser::Expression* expr, size_t row_idx) const;

private:
    // Evaluate specific expression types
    Value evaluateColumnExpression(const parser::ColumnExpression* expr, size_t row_idx) const;
    Value evaluateConstantExpression(const parser::ConstantExpression* expr) const;
    Value evaluateBinaryExpression(const parser::BinaryExpression* expr, size_t row_idx) const;
    Value evaluateLogicalExpression(const parser::LogicalExpression* expr, size_t row_idx) const;
    Value evaluateAggregateExpression(const parser::AggregateExpression* expr, size_t row_idx) const;

    // Helper functions for binary operations
    Value performBinaryOperation(const Value& left, const Value& right, parser::OperatorType op) const;

    // Type checking and conversion helpers
    bool isNumeric(const Value& value) const;
    double toDouble(const Value& value) const;
    bool toBool(const Value& value) const;

    std::shared_ptr<storage::Table> table_;
    std::shared_ptr<storage::Schema> schema_;
};

} // namespace execution
} // namespace gpu_dbms