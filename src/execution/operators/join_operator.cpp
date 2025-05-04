#include "execution/operators/join_operator.hpp"
#include <stdexcept>
#include <variant>

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

// New explicit methods for setting inputs
void JoinOperator::setLeftInput(std::shared_ptr<Result> input) {
    left_input_ = input;
}

void JoinOperator::setRightInput(std::shared_ptr<Result> input) {
    right_input_ = input;
}

// Methods to retrieve table references
const parser::TableRef& JoinOperator::getLeftTableRef() const {
    return left_table_;
}

const parser::TableRef& JoinOperator::getRightTableRef() const {
    return right_table_;
}

std::shared_ptr<Result> JoinOperator::execute() {
    if (!left_input_ || !right_input_ || !left_input_->getData() || !right_input_->getData()) {
        throw std::runtime_error("JoinOperator: Invalid or missing input data");
    }

    auto left_table = left_input_->getData();
    auto right_table = right_input_->getData();
    auto result = std::make_shared<Result>(output_schema_);
    auto result_table = std::make_shared<storage::Table>(output_schema_);

    // Initialize result table columns
    for (const auto& col_info : output_schema_->getColumns()) {
        storage::ColumnData column_data;
        switch (col_info.type) {
            case storage::DataType::INT:
                column_data = storage::IntColumn();
                break;
            case storage::DataType::FLOAT:
                column_data = storage::FloatColumn();
                break;
            case storage::DataType::STRING:
                column_data = storage::StringColumn();
                break;
            case storage::DataType::BOOLEAN:
                column_data = storage::BoolColumn();
                break;
            default:
                throw std::runtime_error("JoinOperator: Unsupported column type for '" + col_info.name + "'");
        }
        result_table->addColumn(col_info.name, column_data);
    }

    // Create a mapping of column names from the schema to column names in the tables
    std::unordered_map<std::string, std::string> left_column_mapping;
    std::unordered_map<std::string, std::string> right_column_mapping;
    
    // Map column names for left table
    for (const auto& col_info : left_schema_->getColumns()) {
        std::string schema_col_name = col_info.name;
        std::string table_col_name;
        
        // Handle prefixed column names
        size_t dot_pos = schema_col_name.find('.');
        if (dot_pos != std::string::npos) {
            table_col_name = schema_col_name.substr(dot_pos + 1);
        } else {
            table_col_name = schema_col_name;
        }
        
        left_column_mapping[schema_col_name] = table_col_name;
    }
    
    // Map column names for right table
    for (const auto& col_info : right_schema_->getColumns()) {
        std::string schema_col_name = col_info.name;
        std::string table_col_name;
        
        // Handle prefixed column names
        size_t dot_pos = schema_col_name.find('.');
        if (dot_pos != std::string::npos) {
            table_col_name = schema_col_name.substr(dot_pos + 1);
        } else {
            table_col_name = schema_col_name;
        }
        
        right_column_mapping[schema_col_name] = table_col_name;
    }

    // Batching parameters
    const size_t batch_size = 1000; // Adjust based on memory constraints
    size_t left_rows = left_table->numRows();

    // Process left table in batches
    for (size_t left_start = 0; left_start < left_rows; left_start += batch_size) {
        size_t left_end = std::min(left_start + batch_size, left_rows);

        for (size_t left_row = left_start; left_row < left_end; ++left_row) {
            for (size_t right_row = 0; right_row < right_table->numRows(); ++right_row) {
                // Create a single-row temporary table for condition evaluation
                auto temp_table = std::make_shared<storage::Table>(output_schema_);
                
                // Add columns from left table to temp table
                for (const auto& col_info : left_schema_->getColumns()) {
                    std::string output_col_name = col_info.name;
                    std::string input_col_name = left_column_mapping[output_col_name];
                    
                    if (!left_table->hasColumn(input_col_name)) {
                        continue;
                    }
                    
                    const auto& src_data = left_table->getColumn(input_col_name);
                    storage::ColumnData temp_data;
                    
                    switch (col_info.type) {
                        case storage::DataType::INT:
                            temp_data = storage::IntColumn(1, std::get<storage::IntColumn>(src_data)[left_row]);
                            break;
                        case storage::DataType::FLOAT:
                            temp_data = storage::FloatColumn(1, std::get<storage::FloatColumn>(src_data)[left_row]);
                            break;
                        case storage::DataType::STRING:
                            temp_data = storage::StringColumn(1, std::get<storage::StringColumn>(src_data)[left_row]);
                            break;
                        case storage::DataType::BOOLEAN:
                            temp_data = storage::BoolColumn(1, std::get<storage::BoolColumn>(src_data)[left_row]);
                            break;
                    }
                    temp_table->addColumn(output_col_name, temp_data);
                }
                
                // Add columns from right table to temp table
                for (const auto& col_info : right_schema_->getColumns()) {
                    std::string output_col_name = col_info.name;
                    std::string input_col_name = right_column_mapping[output_col_name];
                    
                    if (!right_table->hasColumn(input_col_name)) {
                        continue;
                    }
                    
                    const auto& src_data = right_table->getColumn(input_col_name);
                    storage::ColumnData temp_data;
                    
                    switch (col_info.type) {
                        case storage::DataType::INT:
                            temp_data = storage::IntColumn(1, std::get<storage::IntColumn>(src_data)[right_row]);
                            break;
                        case storage::DataType::FLOAT:
                            temp_data = storage::FloatColumn(1, std::get<storage::FloatColumn>(src_data)[right_row]);
                            break;
                        case storage::DataType::STRING:
                            temp_data = storage::StringColumn(1, std::get<storage::StringColumn>(src_data)[right_row]);
                            break;
                        case storage::DataType::BOOLEAN:
                            temp_data = storage::BoolColumn(1, std::get<storage::BoolColumn>(src_data)[right_row]);
                            break;
                    }
                    temp_table->addColumn(output_col_name, temp_data);
                }

                // Evaluate join condition
                ExpressionEvaluator evaluator(temp_table);
                Value condition_result = evaluator.evaluate(condition_.get(), 0);
                
                if (std::holds_alternative<bool>(condition_result) && std::get<bool>(condition_result)) {
                    // Matched rows, add to result
                    const size_t current_result_row = result_table->numRows();
                    
                    // Map of output column name to source type and table
                    for (const auto& col_info : output_schema_->getColumns()) {
                        std::string output_col_name = col_info.name;
                        bool left_has_column = false;
                        bool right_has_column = false;
                        std::string left_col_name;
                        std::string right_col_name;
                        
                        // Check if column comes from left table
                        auto left_it = left_column_mapping.find(output_col_name);
                        if (left_it != left_column_mapping.end()) {
                            left_has_column = left_table->hasColumn(left_it->second);
                            left_col_name = left_it->second;
                        }
                        
                        // Check if column comes from right table
                        auto right_it = right_column_mapping.find(output_col_name);
                        if (right_it != right_column_mapping.end()) {
                            right_has_column = right_table->hasColumn(right_it->second);
                            right_col_name = right_it->second;
                        }
                        
                        // Determine which table to get the data from
                        // Prioritize left table if column name exists in both
                        if (left_has_column) {
                            const auto& src = left_table->getColumn(left_col_name);
                            switch (col_info.type) {
                                case storage::DataType::INT: {
                                    auto& dst = std::get<storage::IntColumn>(result_table->getColumn(output_col_name));
                                    dst.push_back(std::get<storage::IntColumn>(src)[left_row]);
                                    break;
                                }
                                case storage::DataType::FLOAT: {
                                    auto& dst = std::get<storage::FloatColumn>(result_table->getColumn(output_col_name));
                                    dst.push_back(std::get<storage::FloatColumn>(src)[left_row]);
                                    break;
                                }
                                case storage::DataType::STRING: {
                                    auto& dst = std::get<storage::StringColumn>(result_table->getColumn(output_col_name));
                                    dst.push_back(std::get<storage::StringColumn>(src)[left_row]);
                                    break;
                                }
                                case storage::DataType::BOOLEAN: {
                                    auto& dst = std::get<storage::BoolColumn>(result_table->getColumn(output_col_name));
                                    dst.push_back(std::get<storage::BoolColumn>(src)[left_row]);
                                    break;
                                }
                            }
                        } else if (right_has_column) {
                            const auto& src = right_table->getColumn(right_col_name);
                            switch (col_info.type) {
                                case storage::DataType::INT: {
                                    auto& dst = std::get<storage::IntColumn>(result_table->getColumn(output_col_name));
                                    dst.push_back(std::get<storage::IntColumn>(src)[right_row]);
                                    break;
                                }
                                case storage::DataType::FLOAT: {
                                    auto& dst = std::get<storage::FloatColumn>(result_table->getColumn(output_col_name));
                                    dst.push_back(std::get<storage::FloatColumn>(src)[right_row]);
                                    break;
                                }
                                case storage::DataType::STRING: {
                                    auto& dst = std::get<storage::StringColumn>(result_table->getColumn(output_col_name));
                                    dst.push_back(std::get<storage::StringColumn>(src)[right_row]);
                                    break;
                                }
                                case storage::DataType::BOOLEAN: {
                                    auto& dst = std::get<storage::BoolColumn>(result_table->getColumn(output_col_name));
                                    dst.push_back(std::get<storage::BoolColumn>(src)[right_row]);
                                    break;
                                }
                            }
                        }
                        // If the column isn't found in either table, leave it as default/null
                    }
                }
            }
        }
    }

    result->setData(result_table);
    return result;
}

} // namespace execution
} // namespace gpu_dbms