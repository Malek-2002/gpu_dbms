#include "execution/expression_evaluator.hpp"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace gpu_dbms {
namespace execution {

ExpressionEvaluator::ExpressionEvaluator(std::shared_ptr<storage::Table> table)
    : table_(table), schema_(table ? table->getSchema() : nullptr) {
    if (!table_ || !schema_) {
        throw std::runtime_error("ExpressionEvaluator: Invalid table or schema");
    }
}

Value ExpressionEvaluator::evaluate(const parser::Expression* expr, size_t row_idx) const {
    if (!expr) {
        throw std::runtime_error("ExpressionEvaluator: Null expression");
    }

    if (row_idx >= table_->numRows()) {
        throw std::runtime_error("ExpressionEvaluator: Row index out of bounds");
    }

    if (const auto* col_expr = dynamic_cast<const parser::ColumnExpression*>(expr)) {
        return evaluateColumnExpression(col_expr, row_idx);
    } else if (const auto* const_expr = dynamic_cast<const parser::ConstantExpression*>(expr)) {
        return evaluateConstantExpression(const_expr);
    } else if (const auto* bin_expr = dynamic_cast<const parser::BinaryExpression*>(expr)) {
        return evaluateBinaryExpression(bin_expr, row_idx);
    } else if (const auto* logical_expr = dynamic_cast<const parser::LogicalExpression*>(expr)) {
        return evaluateLogicalExpression(logical_expr, row_idx);
    } else if (const auto* agg_expr = dynamic_cast<const parser::AggregateExpression*>(expr)) {
        return evaluateAggregateExpression(agg_expr, row_idx);
    } else {
        throw std::runtime_error("ExpressionEvaluator: Unknown expression type");
    }
}

Value ExpressionEvaluator::evaluateColumnExpression(const parser::ColumnExpression* expr, size_t row_idx) const {
    if (!expr) {
        throw std::runtime_error("ExpressionEvaluator: Null ColumnExpression");
    }

    std::string col_name = expr->column.column_name;

    try {
        const auto& col_info = schema_->getColumn(col_name);
        const auto& col_data = table_->getColumn(col_name);

        switch (col_info.type) {
            case storage::DataType::INT:
                return std::get<storage::IntColumn>(col_data)[row_idx];
            case storage::DataType::FLOAT:
                return std::get<storage::FloatColumn>(col_data)[row_idx];
            case storage::DataType::STRING:
                return std::get<storage::StringColumn>(col_data)[row_idx];
            case storage::DataType::BOOLEAN:
                return std::get<storage::BoolColumn>(col_data)[row_idx];
            default:
                throw std::runtime_error("Unsupported column type for column: " + col_name);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("ExpressionEvaluator: Failed to access column '" + col_name + "': " + e.what());
    }
}

Value ExpressionEvaluator::evaluateConstantExpression(const parser::ConstantExpression* expr) const {
    if (!expr) {
        throw std::runtime_error("ExpressionEvaluator: Null ConstantExpression");
    }

    switch (expr->type) {
        case parser::ConstantExpression::ValueType::INTEGER:
            return expr->int_value;
        case parser::ConstantExpression::ValueType::FLOAT:
            return expr->float_value;
        case parser::ConstantExpression::ValueType::STRING:
            return expr->string_value;
        case parser::ConstantExpression::ValueType::BOOLEAN:
            return expr->bool_value;
        case parser::ConstantExpression::ValueType::NULL_VALUE:
            return std::monostate{};
        default:
            throw std::runtime_error("ExpressionEvaluator: Unsupported constant type");
    }
}

Value ExpressionEvaluator::evaluateBinaryExpression(const parser::BinaryExpression* expr, size_t row_idx) const {
    if (!expr || !expr->left || !expr->right) {
        throw std::runtime_error("ExpressionEvaluator: Invalid BinaryExpression");
    }

    Value left_val = evaluate(expr->left.get(), row_idx);
    Value right_val = evaluate(expr->right.get(), row_idx);

    return performBinaryOperation(left_val, right_val, expr->op);
}

Value ExpressionEvaluator::evaluateLogicalExpression(const parser::LogicalExpression* expr, size_t row_idx) const {
    if (!expr || !expr->left || !expr->right) {
        throw std::runtime_error("ExpressionEvaluator: Invalid LogicalExpression");
    }

    Value left_val = evaluate(expr->left.get(), row_idx);
    Value right_val = evaluate(expr->right.get(), row_idx);

    bool left_bool = toBool(left_val);
    bool right_bool = toBool(right_val);

    switch (expr->op) {
        case parser::LogicalOperatorType::AND:
            return left_bool && right_bool;
        case parser::LogicalOperatorType::OR:
            return left_bool || right_bool;
        default:
            throw std::runtime_error("ExpressionEvaluator: Unsupported logical operator");
    }
}

Value ExpressionEvaluator::evaluateAggregateExpression(const parser::AggregateExpression* expr, size_t row_idx) const {
    if (!expr) {
        throw std::runtime_error("ExpressionEvaluator: Null AggregateExpression");
    }

    // Get the entire column data for the expression
    if (auto col_expr = dynamic_cast<const parser::ColumnExpression*>(expr->expr.get())) {
        std::string col_name = col_expr->column.column_name;

        try {
            const auto& col_info = schema_->getColumn(col_name);
            const auto& col_data = table_->getColumn(col_name);
            
            // Perform the aggregation based on the type
            switch (expr->type) {
                case parser::AggregateType::COUNT: {
                    // Just return the size of the column
                    return static_cast<int64_t>(table_->numRows());
                }
                
                case parser::AggregateType::SUM: {
                    if (std::holds_alternative<storage::IntColumn>(col_data)) {
                        const auto& data = std::get<storage::IntColumn>(col_data);
                        int64_t sum = std::accumulate(data.begin(), data.end(), static_cast<int64_t>(0));
                        return sum;
                    } else if (std::holds_alternative<storage::FloatColumn>(col_data)) {
                        const auto& data = std::get<storage::FloatColumn>(col_data);
                        double sum = std::accumulate(data.begin(), data.end(), 0.0);
                        return sum;
                    } else if (std::holds_alternative<storage::BoolColumn>(col_data)) {
                        const auto& data = std::get<storage::BoolColumn>(col_data);
                        int64_t sum = std::count(data.begin(), data.end(), true);
                        return sum;
                    } else {
                        throw std::runtime_error("ExpressionEvaluator: SUM not supported for this column type");
                    }
                }
                
                case parser::AggregateType::AVG: {
                    if (std::holds_alternative<storage::IntColumn>(col_data)) {
                        const auto& data = std::get<storage::IntColumn>(col_data);
                        if (data.empty()) {
                            return 0.0;
                        }
                        double sum = std::accumulate(data.begin(), data.end(), 0.0);
                        return sum / data.size();
                    } else if (std::holds_alternative<storage::FloatColumn>(col_data)) {
                        const auto& data = std::get<storage::FloatColumn>(col_data);
                        if (data.empty()) {
                            return 0.0;
                        }
                        double sum = std::accumulate(data.begin(), data.end(), 0.0);
                        return sum / data.size();
                    } else if (std::holds_alternative<storage::BoolColumn>(col_data)) {
                        const auto& data = std::get<storage::BoolColumn>(col_data);
                        if (data.empty()) {
                            return 0.0;
                        }
                        double sum = std::count(data.begin(), data.end(), true);
                        return sum / data.size();
                    } else {
                        throw std::runtime_error("ExpressionEvaluator: AVG not supported for this column type");
                    }
                }
                
                case parser::AggregateType::MIN: {
                    if (std::holds_alternative<storage::IntColumn>(col_data)) {
                        const auto& data = std::get<storage::IntColumn>(col_data);
                        if (data.empty()) {
                            return std::numeric_limits<int64_t>::max();
                        }
                        return *std::min_element(data.begin(), data.end());
                    } else if (std::holds_alternative<storage::FloatColumn>(col_data)) {
                        const auto& data = std::get<storage::FloatColumn>(col_data);
                        if (data.empty()) {
                            return std::numeric_limits<double>::max();
                        }
                        return *std::min_element(data.begin(), data.end());
                    } else if (std::holds_alternative<storage::StringColumn>(col_data)) {
                        const auto& data = std::get<storage::StringColumn>(col_data);
                        if (data.empty()) {
                            return std::string();
                        }
                        return *std::min_element(data.begin(), data.end());
                    } else if (std::holds_alternative<storage::BoolColumn>(col_data)) {
                        const auto& data = std::get<storage::BoolColumn>(col_data);
                        if (data.empty()) {
                            return true;
                        }
                        return *std::min_element(data.begin(), data.end());
                    } else {
                        throw std::runtime_error("ExpressionEvaluator: MIN not supported for this column type");
                    }
                }
                
                case parser::AggregateType::MAX: {
                    if (std::holds_alternative<storage::IntColumn>(col_data)) {
                        const auto& data = std::get<storage::IntColumn>(col_data);
                        if (data.empty()) {
                            return std::numeric_limits<int64_t>::lowest();
                        }
                        return *std::max_element(data.begin(), data.end());
                    } else if (std::holds_alternative<storage::FloatColumn>(col_data)) {
                        const auto& data = std::get<storage::FloatColumn>(col_data);
                        if (data.empty()) {
                            return std::numeric_limits<double>::lowest();
                        }
                        return *std::max_element(data.begin(), data.end());
                    } else if (std::holds_alternative<storage::StringColumn>(col_data)) {
                        const auto& data = std::get<storage::StringColumn>(col_data);
                        if (data.empty()) {
                            return std::string();
                        }
                        return *std::max_element(data.begin(), data.end());
                    } else if (std::holds_alternative<storage::BoolColumn>(col_data)) {
                        const auto& data = std::get<storage::BoolColumn>(col_data);
                        if (data.empty()) {
                            return false;
                        }
                        return *std::max_element(data.begin(), data.end());
                    } else {
                        throw std::runtime_error("ExpressionEvaluator: MAX not supported for this column type");
                    }
                }
                
                default:
                    throw std::runtime_error("ExpressionEvaluator: Unsupported aggregate function");
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("ExpressionEvaluator: Failed to access column '" + col_name + "': " + e.what());
        }
    } else {
        // Handle more complex expressions for aggregation
        // This would require evaluating the expression for each row and then aggregating
        throw std::runtime_error("ExpressionEvaluator: Complex expressions in aggregates not yet supported");
    }
}

Value ExpressionEvaluator::performBinaryOperation(const Value& left, const Value& right, parser::OperatorType op) const {
    // Handle NULL values
    if (std::holds_alternative<std::monostate>(left) || std::holds_alternative<std::monostate>(right)) {
        return std::monostate{};
    }

    // Numeric operations
    if (isNumeric(left) && isNumeric(right)) {
        double l = toDouble(left);
        double r = toDouble(right);

        switch (op) {
            case parser::OperatorType::EQUALS:
                return std::abs(l - r) < 1e-10; // Floating-point comparison
            case parser::OperatorType::NOT_EQUALS:
                return std::abs(l - r) >= 1e-10;
            case parser::OperatorType::LESS_THAN:
                return l < r;
            case parser::OperatorType::GREATER_THAN:
                return l > r;
            case parser::OperatorType::LESS_EQUALS:
                return l <= r;
            case parser::OperatorType::GREATER_EQUALS:
                return l >= r;
            default:
                throw std::runtime_error("ExpressionEvaluator: Unsupported binary operator for numeric types");
        }
    }

    // String comparison
    if (std::holds_alternative<std::string>(left) && std::holds_alternative<std::string>(right)) {
        const auto& l = std::get<std::string>(left);
        const auto& r = std::get<std::string>(right);

        switch (op) {
            case parser::OperatorType::EQUALS:
                return l == r;
            case parser::OperatorType::NOT_EQUALS:
                return l != r;
            case parser::OperatorType::LESS_THAN:
                return l < r;
            case parser::OperatorType::GREATER_THAN:
                return l > r;
            case parser::OperatorType::LESS_EQUALS:
                return l <= r;
            case parser::OperatorType::GREATER_EQUALS:
                return l >= r;
            default:
                throw std::runtime_error("ExpressionEvaluator: Unsupported binary operator for strings");
        }
    }

    // Boolean comparison
    if (std::holds_alternative<bool>(left) && std::holds_alternative<bool>(right)) {
        bool l = std::get<bool>(left);
        bool r = std::get<bool>(right);

        switch (op) {
            case parser::OperatorType::EQUALS:
                return l == r;
            case parser::OperatorType::NOT_EQUALS:
                return l != r;
            default:
                throw std::runtime_error("ExpressionEvaluator: Unsupported binary operator for booleans");
        }
    }

    throw std::runtime_error("ExpressionEvaluator: Incompatible types for binary operation");
}

bool ExpressionEvaluator::isNumeric(const Value& value) const {
    return std::holds_alternative<int64_t>(value) || std::holds_alternative<double>(value);
}

double ExpressionEvaluator::toDouble(const Value& value) const {
    if (std::holds_alternative<int64_t>(value)) {
        return static_cast<double>(std::get<int64_t>(value));
    } else if (std::holds_alternative<double>(value)) {
        return std::get<double>(value);
    }
    throw std::runtime_error("ExpressionEvaluator: Cannot convert to double");
}

bool ExpressionEvaluator::toBool(const Value& value) const {
    if (std::holds_alternative<bool>(value)) {
        return std::get<bool>(value);
    } else if (std::holds_alternative<std::monostate>(value)) {
        return false; // NULL treated as false in logical operations
    }
    throw std::runtime_error("ExpressionEvaluator: Cannot convert to boolean");
}

} // namespace execution
} // namespace gpu_dbms