#include "execution/operators/join_operator.hpp"
#include "storage/table.hpp"
#include "storage/schema.hpp"
#include <stdexcept>
#include <vector>
#include <variant>

extern std::vector<int> launch_join_compare(
    const std::vector<int>& left,
    const std::vector<int>& right,
    size_t n, size_t m,
    int op_code);

extern std::vector<int> launch_join_compare(
    const std::vector<long>& left,
    const std::vector<long>& right,
    size_t n, size_t m,
    int op_code);

extern std::vector<int> launch_join_compare(
    const std::vector<double>& left,
    const std::vector<double>& right,
    size_t n, size_t m,
    int op_code);

extern std::vector<int> launch_join_and_or(
    const std::vector<int>& left,
    const std::vector<int>& right,
    size_t n, size_t m,
    int op_code);

namespace gpu_dbms {
namespace execution {

JoinOperator::JoinOperator(
    storage::Catalog& catalog,
    const parser::TableRef& left_table,
    const parser::TableRef& right_table,
    std::shared_ptr<parser::Expression> condition,
    std::shared_ptr<storage::Schema> left_schema,
    std::shared_ptr<storage::Schema> right_schema,
    std::shared_ptr<storage::Schema> output_schema
) : catalog_(catalog),
    left_table_(left_table),
    right_table_(right_table),
    condition_(condition),
    left_schema_(left_schema),
    right_schema_(right_schema),
    output_schema_(output_schema),
    left_input_(nullptr),
    right_input_(nullptr) {
    if (!condition_ || !left_schema_ || !right_schema_ || !output_schema_) {
        throw std::runtime_error("JoinOperator: Invalid initialization parameters");
    }
}

std::shared_ptr<storage::Schema> JoinOperator::getOutputSchema() const {
    return output_schema_;
}

void JoinOperator::setInput(std::shared_ptr<Result> input) {
    if (!left_input_) {
        left_input_ = input;
    } else if (!right_input_) {
        right_input_ = input;
    } else {
        throw std::runtime_error("JoinOperator: Both inputs already set");
    }
}

void JoinOperator::setLeftInput(std::shared_ptr<Result> input) {
    left_input_ = input;
}

void JoinOperator::setRightInput(std::shared_ptr<Result> input) {
    right_input_ = input;
}

const parser::TableRef& JoinOperator::getLeftTableRef() const {
    return left_table_;
}

const parser::TableRef& JoinOperator::getRightTableRef() const {
    return right_table_;
}


void evaluateBinaryExpression(
    const parser::BinaryExpression& expr,
    const storage::Table& left_table,
    const storage::Table& right_table,
    const parser::TableRef& left_table_ref,
    const parser::TableRef& right_table_ref,
    std::vector<int>& result) {

    // Extract column expressions
    auto* left_col_expr = dynamic_cast<parser::ColumnExpression*>(expr.left.get());
    auto* right_col_expr = dynamic_cast<parser::ColumnExpression*>(expr.right.get());

    if (!left_col_expr || !right_col_expr) {
        throw std::runtime_error("Invalid column expression in binary expression");
    }

    const std::string& left_col_name = left_col_expr->column.column_name;
    const std::string& right_col_name = right_col_expr->column.column_name;

    const std::string& left_col_table_name = left_col_expr->column.table_name;
    const std::string& right_col_table_name = right_col_expr->column.table_name;

    const storage::Table* actual_left_table = nullptr;
    const storage::Table* actual_right_table = nullptr;

    // Identify which table each column comes from
    if (left_col_table_name == left_table_ref.table_name) {
        actual_left_table = &left_table;
    } else if (left_col_table_name == right_table_ref.alias) {
        actual_left_table = &right_table;
    } else {
        throw std::runtime_error("Left column alias does not match any table alias");
    }

    if (right_col_table_name == right_table_ref.table_name) {
        actual_right_table = &right_table;
    } else if (right_col_table_name == left_table_ref.alias) {
        actual_right_table = &left_table;
    } else {
        throw std::runtime_error("Right column alias does not match any table alias");
    }

    const auto& left_data_variant = actual_left_table->getColumn(left_col_name);
    const auto& right_data_variant = actual_right_table->getColumn(right_col_name);

    size_t n = left_table.numRows();
    size_t m = right_table.numRows();
    auto op = expr.op;

    if (std::holds_alternative<storage::IntColumn>(left_data_variant)) {
        const auto& left_data = std::get<storage::IntColumn>(left_data_variant);
        const auto& right_data = std::get<storage::IntColumn>(right_data_variant);

        result = launch_join_compare(left_data, right_data, n, m, static_cast<int>(op));

        // for (size_t i = 0; i < n; ++i) {
        //     for (size_t j = 0; j < m; ++j) {
        //         switch (op) {
        //             case parser::OperatorType::EQUALS:         result[i][j] = (left_data[i] == right_data[j]); break;
        //             case parser::OperatorType::NOT_EQUALS:     result[i][j] = (left_data[i] != right_data[j]); break;
        //             case parser::OperatorType::LESS_THAN:      result[i][j] = (left_data[i] <  right_data[j]); break;
        //             case parser::OperatorType::GREATER_THAN:   result[i][j] = (left_data[i] >  right_data[j]); break;
        //             case parser::OperatorType::LESS_EQUALS:    result[i][j] = (left_data[i] <= right_data[j]); break;
        //             case parser::OperatorType::GREATER_EQUALS: result[i][j] = (left_data[i] >= right_data[j]); break;
        //         }
        //     }
        // }
    } else if (std::holds_alternative<storage::FloatColumn>(left_data_variant)) {
        const auto& left_data = std::get<storage::FloatColumn>(left_data_variant);
        const auto& right_data = std::get<storage::FloatColumn>(right_data_variant);

        result = launch_join_compare(left_data, right_data, n, m, static_cast<int>(op));

        // for (size_t i = 0; i < n; ++i) {
        //     for (size_t j = 0; j < m; ++j) {
        //         switch (op) {
        //             case parser::OperatorType::EQUALS:         result[i][j] = (left_data[i] == right_data[j]); break;
        //             case parser::OperatorType::NOT_EQUALS:     result[i][j] = (left_data[i] != right_data[j]); break;
        //             case parser::OperatorType::LESS_THAN:      result[i][j] = (left_data[i] <  right_data[j]); break;
        //             case parser::OperatorType::GREATER_THAN:   result[i][j] = (left_data[i] >  right_data[j]); break;
        //             case parser::OperatorType::LESS_EQUALS:    result[i][j] = (left_data[i] <= right_data[j]); break;
        //             case parser::OperatorType::GREATER_EQUALS: result[i][j] = (left_data[i] >= right_data[j]); break;
        //         }
        //     }
        // }
    } else if (std::holds_alternative<storage::StringColumn>(left_data_variant)) {
        const auto& left_data = std::get<storage::StringColumn>(left_data_variant);
        const auto& right_data = std::get<storage::StringColumn>(right_data_variant);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                switch (op) {
                    case parser::OperatorType::EQUALS:         result[i * m + j] = (left_data[i] == right_data[j]); break;
                    // case parser::OperatorType::EQUALS:     result[i][j] = (left_data[i] == right_data[j]); break;
                    // case parser::OperatorType::NOT_EQUALS: result[i][j] = (left_data[i] != right_data[j]); break;
                    case parser::OperatorType::NOT_EQUALS:     result[i * m + j] = (left_data[i] != right_data[j]); break;
                    default: break; // skip invalid ops
                }
            }
        }
    } else if (std::holds_alternative<storage::BoolColumn>(left_data_variant)) {
        const auto& left_data = std::get<storage::BoolColumn>(left_data_variant);
        const auto& right_data = std::get<storage::BoolColumn>(right_data_variant);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                switch (op) {
                    case parser::OperatorType::EQUALS:     result[i * m + j] = (left_data[i] == right_data[j]); break;
                    case parser::OperatorType::NOT_EQUALS: result[i * m + j] = (left_data[i] != right_data[j]); break;
                    default: break; // optional for <, > etc.
                }
            }
        }
    }
}


// Helper function to evaluate an expression recursively
void evaluateExpression(
    const std::shared_ptr<parser::Expression>& expr,
    const storage::Table& left_table,
    const storage::Table& right_table,
    const parser::TableRef& left_table_ref,
    const parser::TableRef& right_table_ref,
    std::vector<int>& result) {
    
    size_t n = left_table.numRows();
    size_t m = right_table.numRows();

    if (auto bin_expr = std::dynamic_pointer_cast<parser::BinaryExpression>(expr)) {
        evaluateBinaryExpression(*bin_expr, left_table, right_table, left_table_ref, right_table_ref, result);
    } else if (auto logical_expr = std::dynamic_pointer_cast<parser::LogicalExpression>(expr)) {
        std::vector<int> left_result(n * m, 0);
        std::vector<int> right_result(n * m, 0);

        evaluateExpression(logical_expr->left, left_table, right_table, left_table_ref, right_table_ref, left_result);
        evaluateExpression(logical_expr->right, left_table, right_table, left_table_ref, right_table_ref, right_result);

        result = launch_join_and_or(left_result, right_result, n, m, static_cast<int>(logical_expr->op));

        // for (size_t i = 0; i < n; ++i) {
        //     for (size_t j = 0; j < m; ++j) {
        //         if (logical_expr->op == parser::LogicalOperatorType::AND) {
        //             result[i * m + j] = left_result[i * m + j] && right_result[i * m + j];
        //             // result[i][j] = left_result[i][j] && right_result[i][j];
        //         } else if (logical_expr->op == parser::LogicalOperatorType::OR) {
        //             result[i * m + j] = left_result[i * m + j] || right_result[i * m + j];
        //             // result[i][j] = left_result[i][j] || right_result[i][j];
        //         }
        //     }
        // }
    } else {
        throw std::runtime_error("Unsupported expression type in join condition");
    }
}


// Helper method to copy a value from source column to result column
void copyColumnValue(const storage::ColumnData& source_col, 
    storage::ColumnData& result_col,
    size_t source_idx, size_t result_idx,
    storage::DataType type) {
    switch (type) {
        case storage::DataType::INT: {
            auto& src = std::get<storage::IntColumn>(source_col);
            auto& dst = std::get<storage::IntColumn>(result_col);
            dst[result_idx] = src[source_idx];
            break;
    }
    case storage::DataType::FLOAT: {
            auto& src = std::get<storage::FloatColumn>(source_col);
            auto& dst = std::get<storage::FloatColumn>(result_col);
            dst[result_idx] = src[source_idx];
            break;
    }
    case storage::DataType::STRING: {
            auto& src = std::get<storage::StringColumn>(source_col);
            auto& dst = std::get<storage::StringColumn>(result_col);
            dst[result_idx] = src[source_idx];
            break;
    }
        case storage::DataType::BOOLEAN: {
        auto& src = std::get<storage::BoolColumn>(source_col);
        auto& dst = std::get<storage::BoolColumn>(result_col);
        dst[result_idx] = src[source_idx];
        break;
    }
    default:
        throw std::runtime_error("JoinOperator: Unsupported column type in copyColumnValue");
    }
}


std::shared_ptr<Result> JoinOperator::execute() {
    if (!left_input_ || !right_input_ || !left_input_->getData() || !right_input_->getData()) {
        throw std::runtime_error("JoinOperator: Invalid or missing input data");
    }

    auto left_table = left_input_->getData();
    auto right_table = right_input_->getData();
    auto result = std::make_shared<Result>(output_schema_);
    auto result_table = std::make_shared<storage::Table>(output_schema_);
    size_t n = left_table->numRows();
    size_t m = right_table->numRows();

    std::vector<int> evaluated_matrix(n * m, 0);

    // Evaluate the condition to get the (n × m) boolean array
    evaluateExpression(condition_, *left_table, *right_table, left_table_, right_table_, evaluated_matrix);

    // Count the number of rows in the result
    size_t result_row_count = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            if (evaluated_matrix[i * m + j]) {
                ++result_row_count;
            }
        }
    }

    // Initialize columns for the result table with proper size
    for (const auto& col_info : output_schema_->getColumns()) {
        const auto& col_name = col_info.name;
        storage::ColumnData output_col;

        switch (col_info.type) {
            case storage::DataType::INT:
                output_col = storage::IntColumn(result_row_count);
                break;
            case storage::DataType::FLOAT:
                output_col = storage::FloatColumn(result_row_count);
                break;
            case storage::DataType::STRING:
                output_col = storage::StringColumn(result_row_count);
                break;
            case storage::DataType::BOOLEAN:
                output_col = storage::BoolColumn(result_row_count);
                break;
            default:
                throw std::runtime_error("JoinOperator: Unsupported column type for column '" + col_name + "'");
        }

        result_table->addColumn(col_name, std::move(output_col));
    }

    // Populate the result table with joined rows
    size_t result_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            if (evaluated_matrix[i * m + j]) {
                // Merge the rows from left and right tables
                for (const auto& col_info : output_schema_->getColumns()) {
                    const auto& col_name = col_info.name;
                    auto& result_col = result_table->getColumn(col_name);

                    // Check if column belongs to left or right table
                    bool is_left_column = left_table->hasColumn(col_name);
                    bool is_right_column = right_table->hasColumn(col_name);

                    if (is_left_column) {
                        const auto& source_col = left_table->getColumn(col_name);
                        copyColumnValue(source_col, result_col, i, result_idx, col_info.type);
                    } else if (is_right_column) {
                        const auto& source_col = right_table->getColumn(col_name);
                        copyColumnValue(source_col, result_col, j, result_idx, col_info.type);
                    } else {
                        throw std::runtime_error("JoinOperator: Column '" + col_name + 
                                               "' not found in either left or right table");
                    }
                }
                ++result_idx;
            }
        }
    }

    result->setData(result_table);
    return result;
}

} // namespace execution
} // namespace gpu_dbms