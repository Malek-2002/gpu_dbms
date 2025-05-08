#include "execution/operators/filter_operator.hpp"
#include "execution/expression_evaluator.hpp"
#include <stdexcept>

namespace gpu_dbms {
namespace execution {

FilterOperator::FilterOperator(storage::Catalog& catalog,
                               const parser::TableRef& table_ref,
                               std::shared_ptr<parser::Expression> filter_expr,
                               std::shared_ptr<storage::Schema> input_schema)
    : catalog_(catalog), table_ref_(table_ref), filter_expr_(filter_expr), input_schema_(input_schema), input_(nullptr) {
    if (!filter_expr_) {
        throw std::runtime_error("FilterOperator: Null filter expression");
    }
    if (!input_schema_) {
        throw std::runtime_error("FilterOperator: Null input schema");
    }
    if (table_ref_.table_name.empty()) {
        throw std::runtime_error("FilterOperator: Empty table name in TableRef");
    }
    if (!catalog_.hasTable(table_ref_.table_name)) {
        throw std::runtime_error("FilterOperator: Table '" + table_ref_.table_name + "' not found in catalog");
    }
}

std::shared_ptr<Result> FilterOperator::execute() {
    std::shared_ptr<storage::Table> input_table;
    
    // If no input is provided, fetch the table from the catalog (source operator)
    if (!input_ || !input_->getData()) {
        input_table = catalog_.getTable(table_ref_.table_name);
        if (!input_table) {
            throw std::runtime_error("FilterOperator: Failed to retrieve table '" + table_ref_.table_name + "'");
        }
    } else {
        input_table = input_->getData();
    }

    // Verify input schema matches
    if (input_table->getSchema() != input_schema_) {
        // throw std::runtime_error("FilterOperator: Input table schema does not match expected schema");
        input_schema_ = input_table->getSchema(); // Update to match the input table schema
    }

    size_t num_rows = input_table->numRows();

    // Create output table with same schema
    auto output_table = input_table->createEmptyTable();
    ExpressionEvaluator evaluator(input_table);

    // Evaluate filter expression for each row
    std::vector<bool> keep_row(num_rows);
    size_t output_rows = 0;

    for (size_t row = 0; row < num_rows; ++row) {
        auto val = evaluator.evaluate(filter_expr_.get(), row);
        if (std::holds_alternative<std::monostate>(val)) {
            keep_row[row] = false; // NULL treated as false in WHERE
        } else if (std::holds_alternative<bool>(val)) {
            keep_row[row] = std::get<bool>(val);
        } else {
            throw std::runtime_error("FilterOperator: Filter expression must evaluate to boolean");
        }
        if (keep_row[row]) {
            ++output_rows;
        }
    }

    // Copy filtered rows to output table
    for (const auto& col_info : input_schema_->getColumns()) {
        const auto& col_name = col_info.name;
        const auto& input_col = input_table->getColumn(col_name);
        storage::ColumnData output_col;

        switch (col_info.type) {
            case storage::DataType::INT: {
                storage::IntColumn data(output_rows);
                size_t out_idx = 0;
                for (size_t row = 0; row < num_rows; ++row) {
                    if (keep_row[row]) {
                        data[out_idx++] = std::get<storage::IntColumn>(input_col)[row];
                    }
                }
                output_col = std::move(data);
                break;
            }
            case storage::DataType::FLOAT: {
                storage::FloatColumn data(output_rows);
                size_t out_idx = 0;
                for (size_t row = 0; row < num_rows; ++row) {
                    if (keep_row[row]) {
                        data[out_idx++] = std::get<storage::FloatColumn>(input_col)[row];
                    }
                }
                output_col = std::move(data);
                break;
            }
            case storage::DataType::STRING: {
                storage::StringColumn data(output_rows);
                size_t out_idx = 0;
                for (size_t row = 0; row < num_rows; ++row) {
                    if (keep_row[row]) {
                        data[out_idx++] = std::get<storage::StringColumn>(input_col)[row];
                    }
                }
                output_col = std::move(data);
                break;
            }
            case storage::DataType::BOOLEAN: {
                storage::BoolColumn data(output_rows);
                size_t out_idx = 0;
                for (size_t row = 0; row < num_rows; ++row) {
                    if (keep_row[row]) {
                        data[out_idx++] = std::get<storage::BoolColumn>(input_col)[row];
                    }
                }
                output_col = std::move(data);
                break;
            }
            default:
                throw std::runtime_error("FilterOperator: Unsupported column type for column '" + col_name + "'");
        }

        output_table->addColumn(col_name, std::move(output_col));
    }

    // Create and return result
    auto result = std::make_shared<Result>(input_schema_);
    result->setData(output_table);
    return result;
}

void FilterOperator::setInput(std::shared_ptr<Result> input) {
    input_ = input;
}

std::shared_ptr<storage::Schema> FilterOperator::getOutputSchema() const {
    return input_schema_;
}

} // namespace execution
} // namespace gpu_dbms