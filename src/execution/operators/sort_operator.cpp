#include "execution/operators/sort_operator.hpp"
#include "execution/expression_evaluator.hpp"
#include "storage/table.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace gpu_dbms {
namespace execution {

SortOperator::SortOperator(
    const std::vector<std::pair<std::shared_ptr<parser::Expression>, parser::SortOrder>>& sort_keys,
    std::shared_ptr<storage::Schema> input_schema)
    : sort_keys_(sort_keys), input_schema_(input_schema), input_(nullptr) {
    if (sort_keys_.empty()) {
        throw std::runtime_error("SortOperator: No sort keys provided");
    }
    if (!input_schema_) {
        throw std::runtime_error("SortOperator: Invalid input schema");
    }
}

std::shared_ptr<storage::Schema> SortOperator::getOutputSchema() const {
    return input_schema_;
}

void SortOperator::setInput(std::shared_ptr<Result> input) {
    input_ = input;
}

std::shared_ptr<Result> SortOperator::execute() {
    if (!input_ || !input_->getData()) {
        throw std::runtime_error("SortOperator: Invalid or missing input data");
    }

    auto input_table = input_->getData();
    if (!input_table) {
        throw std::runtime_error("SortOperator: Input table is null");
    }

    // Create result table with the same schema
    auto result = std::make_shared<Result>(input_schema_);
    auto result_table = std::make_shared<storage::Table>(input_schema_);

    // Initialize result table columns
    for (const auto& col_info : input_schema_->getColumns()) {
        const auto& col_name = col_info.name;
        storage::ColumnData output_col;

        switch (col_info.type) {
            case storage::DataType::INT:
                output_col = storage::IntColumn(input_table->numRows());
                break;
            case storage::DataType::FLOAT:
                output_col = storage::FloatColumn(input_table->numRows());
                break;
            case storage::DataType::STRING:
                output_col = storage::StringColumn(input_table->numRows());
                break;
            case storage::DataType::BOOLEAN:
                output_col = storage::BoolColumn(input_table->numRows());
                break;
            default:
                throw std::runtime_error("SortOperator: Unsupported column type for column '" + col_name + "'");
        }

        result_table->addColumn(col_name, std::move(output_col));
    }

    // Create an index vector for sorting
    std::vector<size_t> indices(input_table->numRows());
    std::iota(indices.begin(), indices.end(), 0);

    // Create an ExpressionEvaluator for the input table
    ExpressionEvaluator evaluator(input_table);

    // Sort indices based on sort keys
    std::stable_sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        for (const auto& [expr, order] : sort_keys_) {
            // Evaluate the sort key expression for both rows
            Value val_i = evaluator.evaluate(expr.get(), i);
            Value val_j = evaluator.evaluate(expr.get(), j);

            // Compare values based on data type
            bool less;
            if (std::holds_alternative<int64_t>(val_i) && std::holds_alternative<int64_t>(val_j)) {
                less = std::get<int64_t>(val_i) < std::get<int64_t>(val_j);
            } else if (std::holds_alternative<double>(val_i) && std::holds_alternative<double>(val_j)) {
                less = std::get<double>(val_i) < std::get<double>(val_j);
            } else if (std::holds_alternative<std::string>(val_i) && std::holds_alternative<std::string>(val_j)) {
                less = std::get<std::string>(val_i) < std::get<std::string>(val_j);
            } else if (std::holds_alternative<bool>(val_i) && std::holds_alternative<bool>(val_j)) {
                less = std::get<bool>(val_i) < std::get<bool>(val_j);
            } else if (std::holds_alternative<std::monostate>(val_i) && !std::holds_alternative<std::monostate>(val_j)) {
                less = true; // NULLs sort first
            } else if (!std::holds_alternative<std::monostate>(val_i) && std::holds_alternative<std::monostate>(val_j)) {
                less = false;
            } else if (std::holds_alternative<std::monostate>(val_i) && std::holds_alternative<std::monostate>(val_j)) {
                less = false; // Equal NULLs
            } else {
                throw std::runtime_error("SortOperator: Incompatible types for comparison");
            }

            // Adjust for sort order
            bool result = (order == parser::SortOrder::ASC) ? less : !less;
            if (val_i != val_j) {
                return result;
            }
        }
        return false; // Equal rows
    });

    // Copy rows to result table in sorted order
    for (size_t idx = 0; idx < indices.size(); ++idx) {
        size_t src_idx = indices[idx];
        for (const auto& col_info : input_schema_->getColumns()) {
            const auto& col_name = col_info.name;
            auto& result_col = result_table->getColumn(col_name);
            const auto& source_col = input_table->getColumn(col_name);

            switch (col_info.type) {
                case storage::DataType::INT: {
                    auto& dst = std::get<storage::IntColumn>(result_col);
                    auto& src = std::get<storage::IntColumn>(source_col);
                    dst[idx] = src[src_idx];
                    break;
                }
                case storage::DataType::FLOAT: {
                    auto& dst = std::get<storage::FloatColumn>(result_col);
                    auto& src = std::get<storage::FloatColumn>(source_col);
                    dst[idx] = src[src_idx];
                    break;
                }
                case storage::DataType::STRING: {
                    auto& dst = std::get<storage::StringColumn>(result_col);
                    auto& src = std::get<storage::StringColumn>(source_col);
                    dst[idx] = src[src_idx];
                    break;
                }
                case storage::DataType::BOOLEAN: {
                    auto& dst = std::get<storage::BoolColumn>(result_col);
                    auto& src = std::get<storage::BoolColumn>(source_col);
                    dst[idx] = src[src_idx];
                    break;
                }
                default:
                    throw std::runtime_error("SortOperator: Unsupported column type in copy");
            }
        }
    }

    result->setData(result_table);
    return result;
}

} // namespace execution
} // namespace gpu_dbms