#include "execution/operators/select_operator.hpp"
#include <stdexcept>

namespace gpu_dbms {
namespace execution {

SelectOperator::SelectOperator(storage::Catalog& catalog,
                               const parser::TableRef& table_ref,
                               std::shared_ptr<storage::Schema> output_schema)
    : catalog_(catalog), table_ref_(table_ref), output_schema_(output_schema), input_(nullptr) {
    if (!output_schema_) {
        throw std::runtime_error("SelectOperator: Null output schema");
    }
    if (table_ref_.table_name.empty()) {
        throw std::runtime_error("SelectOperator: Empty table name in TableRef");
    }
    if (!catalog_.hasTable(table_ref_.table_name)) {
        throw std::runtime_error("SelectOperator: Table '" + table_ref_.table_name + "' not found in catalog");
    }
}

std::shared_ptr<Result> SelectOperator::execute() {
    // Retrieve table from catalog
    auto table = catalog_.getTable(table_ref_.table_name);
    if (!table) {
        throw std::runtime_error("SelectOperator: Failed to retrieve table '" + table_ref_.table_name + "'");
    }

    // Verify schema compatibility
    auto table_schema = table->getSchema();
    for (const auto& out_col : output_schema_->getColumns()) {
        bool found = false;
        for (const auto& in_col : table_schema->getColumns()) {
            if (out_col.name == in_col.name && out_col.type == in_col.type) {
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("SelectOperator: Output schema column '" + out_col.name + "' not found in table schema or type mismatch");
        }
    }

    // Create output table with output schema
    auto output_table = std::make_shared<storage::Table>(output_schema_);
    size_t num_rows = table->numRows();

    // Copy selected columns from input table to output table
    for (const auto& col_info : output_schema_->getColumns()) {
        const auto& col_name = col_info.name;
        const auto& input_col = table->getColumn(col_name);
        storage::ColumnData output_col;

        switch (col_info.type) {
            case storage::DataType::INT:
                output_col = std::get<storage::IntColumn>(input_col);
                break;
            case storage::DataType::FLOAT:
                output_col = std::get<storage::FloatColumn>(input_col);
                break;
            case storage::DataType::STRING:
                output_col = std::get<storage::StringColumn>(input_col);
                break;
            case storage::DataType::BOOLEAN:
                output_col = std::get<storage::BoolColumn>(input_col);
                break;
            default:
                throw std::runtime_error("SelectOperator: Unsupported column type for column '" + col_name + "'");
        }

        output_table->addColumn(col_name, std::move(output_col));
    }

    // Create and return result
    auto result = std::make_shared<Result>(output_schema_);
    result->setData(output_table);
    return result;
}

void SelectOperator::setInput(std::shared_ptr<Result> input) {
    // SelectOperator is a source operator, input is ignored
    input_ = input; // Store for completeness, but not used
}

std::shared_ptr<storage::Schema> SelectOperator::getOutputSchema() const {
    return output_schema_;
}

} // namespace execution
} // namespace gpu_dbms