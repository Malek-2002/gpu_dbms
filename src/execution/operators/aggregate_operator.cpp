#include "execution/operators/aggregate_operator.hpp"
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <iostream>

// include extern function that calls CUDA kernel
/////////////////// sum kernels //////////////////////
// int version
extern int parallel_sum(const std::vector<int>& data);
// long version
extern long parallel_sum(const std::vector<long>& data);
// double version
extern double parallel_sum(const std::vector<double>& data);
/////////////////// max kernels //////////////////////
// int version
extern int parallel_max(const std::vector<int>& data);
// long version
extern long parallel_max(const std::vector<long>& data);
// double version
extern double parallel_max(const std::vector<double>& data);
// string version
extern std::string parallel_max(const std::vector<std::string>& data);
/////////////////// min kernels ///////////////////////
// int version
extern int parallel_min(const std::vector<int>& data);
// long version
extern long parallel_min(const std::vector<long>& data);
// double version
extern double parallel_min(const std::vector<double>& data);
// string version
extern std::string parallel_min(const std::vector<std::string>& data);

// Explicit instantiations
extern int parallel_reduce_sum(const std::vector<int>&);
extern long parallel_reduce_sum(const std::vector<long>&);
extern double parallel_reduce_sum(const std::vector<double>&);

extern long parallel_reduce_min(const std::vector<long>&);
extern int parallel_reduce_min(const std::vector<int>&);
extern double parallel_reduce_min(const std::vector<double>&);

extern long parallel_reduce_max(const std::vector<long>&);
extern int parallel_reduce_max(const std::vector<int>&);
extern double parallel_reduce_max(const std::vector<double>&);

extern double parallel_reduce_avg(const std::vector<long>&);
extern double parallel_reduce_avg(const std::vector<int>&);
extern double parallel_reduce_avg(const std::vector<float>&);

namespace gpu_dbms {
namespace execution {

AggregateOperator::AggregateOperator(
    storage::Catalog& catalog,
    std::vector<std::shared_ptr<parser::Expression>> exprs,
    std::shared_ptr<storage::Schema> input_schema,
    std::shared_ptr<storage::Schema> output_schema)
    : catalog_(catalog),
      input_schema_(input_schema),
      output_schema_(output_schema) {
    
    // Filter out and store aggregate expressions
    for (const auto& expr : exprs) {
        auto agg_expr = std::dynamic_pointer_cast<parser::AggregateExpression>(expr);
        if (agg_expr) {
            agg_exprs_.push_back(agg_expr);
        }
    }
}

void AggregateOperator::setInput(std::shared_ptr<Result> input) {
    input_ = input;
}

std::shared_ptr<storage::Schema> AggregateOperator::getOutputSchema() const {
    return output_schema_;
}

std::shared_ptr<Result> AggregateOperator::execute() {
    if (!input_) {
        // If no input is provided, fetch data from catalog (source operator case)
        if (table_ref_.table_name.empty()) {
            throw std::runtime_error("AggregateOperator: No input provided and no valid table reference");
        }
        // Fetch table data from catalog
        auto table = catalog_.getTable(table_ref_.table_name);
        if (!table) {
            throw std::runtime_error("AggregateOperator: Table '" + table_ref_.table_name + "' not found in catalog");
        }
        input_ = std::make_shared<Result>(input_schema_);
        input_->setData(table);
    }

    // Create an empty result with the output schema
    auto result = std::make_shared<Result>(output_schema_);
    auto output_table = std::make_shared<storage::Table>(output_schema_);
    
    // Get input data
    auto input_table = input_->getData();
    if (input_table->numRows() == 0) {
        // Handle empty input table case - return empty result
        result->setData(output_table);
        return result;
    }

    // Create expression evaluator for the input table
    ExpressionEvaluator evaluator(input_table);

    // For each aggregate expression, compute the aggregate value
    for (size_t i = 0; i < agg_exprs_.size(); ++i) {
        const auto& agg_expr = agg_exprs_[i];
        // Get target column information
        auto column_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(agg_expr->expr);
        
        // Handle COUNT(*) case
        if (!column_expr && agg_expr->type == parser::AggregateType::COUNT) {
            Value agg_result = static_cast<int64_t>(input_table->numRows());
            const auto& output_column_name = output_schema_->getColumns()[i].name;
            storage::IntColumn col_data = {static_cast<int>(std::get<int64_t>(agg_result))};
            output_table->addColumn(output_column_name, col_data);
            continue;
        }

        if (column_expr) {
            const std::string& column_name = column_expr->column.column_name;
            
            // Get column data
            const auto& column_data = input_table->getColumn(column_name);
            
            // Compute aggregation
            Value agg_result;
            
            switch (agg_expr->type) {
                case parser::AggregateType::COUNT:
                    agg_result = computeCount(column_data);
                    break;
                case parser::AggregateType::SUM:
                    agg_result = computeSum(column_data);
                    break;
                case parser::AggregateType::AVG:
                    agg_result = computeAvg(column_data);
                    break;
                case parser::AggregateType::MIN:
                    agg_result = computeMin(column_data);
                    break;
                case parser::AggregateType::MAX:
                    agg_result = computeMax(column_data);
                    break;
                default:
                    throw std::runtime_error("Unsupported aggregate type");
            }
            
            // Use the column name from output_schema_
            const auto& output_column_name = output_schema_->getColumns()[i].name;
            
            // Create column data based on the aggregation result type
            storage::ColumnData output_column_data;
            
            if (std::holds_alternative<int64_t>(agg_result)) {
                storage::IntColumn col_data = {static_cast<int>(std::get<int64_t>(agg_result))};
                output_column_data = col_data;
            } else if (std::holds_alternative<double>(agg_result)) {
                storage::FloatColumn col_data = {std::get<double>(agg_result)};
                output_column_data = col_data;
            } else if (std::holds_alternative<std::string>(agg_result)) {
                storage::StringColumn col_data = {std::get<std::string>(agg_result)};
                output_column_data = col_data;
            } else if (std::holds_alternative<bool>(agg_result)) {
                storage::BoolColumn col_data = {std::get<bool>(agg_result)};
                output_column_data = col_data;
            } else {
                // Handle null/monostate case
                storage::IntColumn col_data = {0};  // Default to 0 for null values
                output_column_data = col_data;
            }
            
            output_table->addColumn(output_column_name, output_column_data);
        } else {
            // Handle complex expressions inside aggregates
            throw std::runtime_error("Complex expressions in aggregates not yet supported");
        }
    }

    result->setData(output_table);
    return result;
}

Value AggregateOperator::computeCount(const storage::ColumnData& col_data) const {
    size_t count = 0;
    
    if (std::holds_alternative<storage::IntColumn>(col_data)) {
        count = std::get<storage::IntColumn>(col_data).size();
    } else if (std::holds_alternative<storage::FloatColumn>(col_data)) {
        count = std::get<storage::FloatColumn>(col_data).size();
    } else if (std::holds_alternative<storage::StringColumn>(col_data)) {
        count = std::get<storage::StringColumn>(col_data).size();
    } else if (std::holds_alternative<storage::BoolColumn>(col_data)) {
        count = std::get<storage::BoolColumn>(col_data).size();
    }
    
    return static_cast<int64_t>(count);
}

Value AggregateOperator::computeSum(const storage::ColumnData& col_data) const {
    if (std::holds_alternative<storage::IntColumn>(col_data)) {
        const auto& data = std::get<storage::IntColumn>(col_data);
        // Accumulate as double to ensure FLOAT output
        // cpu version
        // double sum = std::accumulate(data.begin(), data.end(), 0.0);
        ///////////////////////////////////////////////////////////////////////
        // gpu version
        // extern function that calls CUDA kernel
        // long sum = parallel_sum(data);
        ///////////////////////////////////////////////////////////////////////
        // streams version
        long sum = parallel_reduce_sum(data);
        return (double)sum;
    } else if (std::holds_alternative<storage::FloatColumn>(col_data)) {
        const auto& data = std::get<storage::FloatColumn>(col_data);
        // cpu version
        // double sum = std::accumulate(data.begin(), data.end(), 0.0);
        ////////////////////////////////////////////////////////////////
        // gpu version
        // extern function that calls CUDA kernel
        // double sum = parallel_sum(data);
        ///////////////////////////////////////////////////////////////////////
        // streams version
        double sum = parallel_reduce_sum(data);
        return sum;
    } else {
        throw std::runtime_error("SUM can only be applied to numeric columns");
    }
}

Value AggregateOperator::computeAvg(const storage::ColumnData& col_data) const {
    if (std::holds_alternative<storage::IntColumn>(col_data)) {
        const auto& data = std::get<storage::IntColumn>(col_data);
        if (data.empty()) return 0.0;
        // cpu version
        // double sum = std::accumulate(data.begin(), data.end(), 0.0);
        ///////////////////////////////////////////////////////////////////////
        // gpu version
        // extern function that calls CUDA kernel
        // long sum = parallel_sum(data);
        ///////////////////////////////////////////////////////////////////////
        // streams version
        long sum = parallel_reduce_sum(data);
        return (double)sum / data.size();
    } else if (std::holds_alternative<storage::FloatColumn>(col_data)) {
        const auto& data = std::get<storage::FloatColumn>(col_data);
        if (data.empty()) return 0.0;
        // cpu version
        // double sum = std::accumulate(data.begin(), data.end(), 0.0);
        ////////////////////////////////////////////////////////////////
        // gpu version
        // extern function that calls CUDA kernel
        // double sum = parallel_sum(data);
        //////////////////////////////////////////////////////////////////
        // streams version
        double sum = parallel_reduce_sum(data);
        return sum / data.size();
    } else {
        throw std::runtime_error("AVG can only be applied to numeric columns");
    }
}
Value AggregateOperator::computeMin(const storage::ColumnData& col_data) const {
    if (std::holds_alternative<storage::IntColumn>(col_data)) {
        const auto& data = std::get<storage::IntColumn>(col_data);
        if (data.empty()) return std::monostate{};
        // cpu version
        // return static_cast<int64_t>(*std::min_element(data.begin(), data.end()));
        ///////////////////////////////////////////////////////////////////////
        // gpu version
        // return static_cast<int64_t>(parallel_min(data));
        ///////////////////////////////////////////////////////////////////////
        // streams version
        return static_cast<int64_t>(parallel_reduce_min(data));
    } else if (std::holds_alternative<storage::FloatColumn>(col_data)) {
        const auto& data = std::get<storage::FloatColumn>(col_data);
        if (data.empty()) return std::monostate{};
        // cpu version
        // return *std::min_element(data.begin(), data.end());
        ////////////////////////////////////////////////////////////////
        // gpu version
        // return parallel_min(data);
        ////////////////////////////////////////////////////////////////
        // streams version
        return parallel_reduce_min(data);
    } else if (std::holds_alternative<storage::StringColumn>(col_data)) {
        const auto& data = std::get<storage::StringColumn>(col_data);
        if (data.empty()) return std::monostate{};
        // cpu version
        // return *std::min_element(data.begin(), data.end());
        ////////////////////////////////////////////////////////////////
        // gpu version
        return parallel_min(data);
    } else {
        throw std::runtime_error("MIN not supported for this column type");
    }
}

Value AggregateOperator::computeMax(const storage::ColumnData& col_data) const {
    if (std::holds_alternative<storage::IntColumn>(col_data)) {
        const auto& data = std::get<storage::IntColumn>(col_data);
        if (data.empty()) return std::monostate{};
        // cpu version
        // return static_cast<int64_t>(*std::max_element(data.begin(), data.end()));
        ///////////////////////////////////////////////////////////////////////
        // gpu version
        // return static_cast<int64_t>(parallel_max(data));
        ///////////////////////////////////////////////////////////////////////
        // streams version
        return static_cast<int64_t>(parallel_reduce_max(data));
    } else if (std::holds_alternative<storage::FloatColumn>(col_data)) {
        const auto& data = std::get<storage::FloatColumn>(col_data);
        if (data.empty()) return std::monostate{};
        // cpu version
        // return *std::max_element(data.begin(), data.end());
        ////////////////////////////////////////////////////////////////
        // gpu version
        // return parallel_max(data);
        ////////////////////////////////////////////////////////////////
        // streams version
        return parallel_reduce_max(data);
    } else if (std::holds_alternative<storage::StringColumn>(col_data)) {
        const auto& data = std::get<storage::StringColumn>(col_data);
        if (data.empty()) return std::monostate{};
        // cpu version
        // return *std::max_element(data.begin(), data.end());
        ////////////////////////////////////////////////////////////////
        // gpu version
        return parallel_max(data);
    } else {
        throw std::runtime_error("MAX not supported for this column type");
    }
}

// Helper method to get string representation of aggregate type
std::string AggregateOperator::getAggregateTypeName(parser::AggregateType type) const {
    switch (type) {
        case parser::AggregateType::COUNT: return "COUNT";
        case parser::AggregateType::SUM: return "SUM";
        case parser::AggregateType::AVG: return "AVG";
        case parser::AggregateType::MIN: return "MIN";
        case parser::AggregateType::MAX: return "MAX";
        default: return "UNKNOWN";
    }
}

// Specialized template implementations for different data types
template<>
Value AggregateOperator::computeAggregation<int>(
    const std::vector<int>& data, parser::AggregateType agg_type) const {
    
    if (data.empty()) {
        return std::monostate{};
    }
    
    switch (agg_type) {
        case parser::AggregateType::COUNT:
            return static_cast<int64_t>(data.size());
        case parser::AggregateType::SUM: {
            // cpu version
            // int64_t sum = std::accumulate(data.begin(), data.end(), 0);
            //////////////////////////////////////////////////////
            // gpu version
            // int sum = parallel_sum(data);
            //////////////////////////////////////////////////////
            // streams version
            int sum = parallel_reduce_sum(data);
            return static_cast<int64_t>(sum);
        }
        case parser::AggregateType::AVG: {
            // cpu version
            // double sum = std::accumulate(data.begin(), data.end(), 0.0);
            //////////////////////////////////////////////////////
            // gpu version
            // int sum = parallel_sum(data);
            //////////////////////////////////////////////////////
            // streams version
            int sum = parallel_reduce_sum(data);
            return (double)sum / data.size();
        }
        case parser::AggregateType::MIN:
            // cpu version
            // return static_cast<int64_t>(*std::min_element(data.begin(), data.end()));
            //////////////////////////////////////////////////////
            // gpu version
            // return static_cast<int64_t>(parallel_min(data));
            //////////////////////////////////////////////////////
            // streams version
            return static_cast<int64_t>(parallel_reduce_min(data));
        case parser::AggregateType::MAX:
            // cpu version
            // return static_cast<int64_t>(*std::max_element(data.begin(), data.end()));
            //////////////////////////////////////////////////////
            // gpu version
            // return static_cast<int64_t>(parallel_max(data));
            //////////////////////////////////////////////////////
            // streams version
            return static_cast<int64_t>(parallel_reduce_max(data));
        default:
            throw std::runtime_error("Unsupported aggregate type");
    }
}

template<>
Value AggregateOperator::computeAggregation<double>(
    const std::vector<double>& data, parser::AggregateType agg_type) const {
    
    if (data.empty()) {
        return std::monostate{};
    }
    
    switch (agg_type) {
        case parser::AggregateType::COUNT:
            return static_cast<int64_t>(data.size());
        case parser::AggregateType::SUM: {
            // cpu version
            // double sum = std::accumulate(data.begin(), data.end(), 0.0);
            //////////////////////////////////////////////////////
            // gpu version
            // double sum = parallel_sum(data);
            //////////////////////////////////////////////////////
            // streams version
            double sum = parallel_reduce_sum(data);
            return sum;
        }
        case parser::AggregateType::AVG: {
            // cpu version
            // double sum = std::accumulate(data.begin(), data.end(), 0.0);
            //////////////////////////////////////////////////////
            // gpu version
            // double sum = parallel_sum(data);
            //////////////////////////////////////////////////////
            // streams version
            double sum = parallel_reduce_sum(data);
            return sum / data.size();
        }
        case parser::AggregateType::MIN:
            // cpu version
            // return *std::min_element(data.begin(), data.end());
            //////////////////////////////////////////////////////
            // gpu version
            // return parallel_min(data);
            //////////////////////////////////////////////////////
            // streams version
            return parallel_reduce_min(data);
        case parser::AggregateType::MAX:
            // cpu version
            // return *std::max_element(data.begin(), data.end());
            //////////////////////////////////////////////////////
            // gpu version
            // return parallel_max(data);
            //////////////////////////////////////////////////////
            // streams version
            return parallel_reduce_max(data);
        default:
            throw std::runtime_error("Unsupported aggregate type");
    }
}

template<>
Value AggregateOperator::computeAggregation<std::string>(
    const std::vector<std::string>& data, parser::AggregateType agg_type) const {
    
    if (data.empty()) {
        return std::monostate{};
    }
    
    switch (agg_type) {
        case parser::AggregateType::COUNT:
            return static_cast<int64_t>(data.size());
        case parser::AggregateType::MIN:
            // cpu version
            // return *std::min_element(data.begin(), data.end());
            //////////////////////////////////////////////////////
            // gpu version
            return parallel_min(data);
        case parser::AggregateType::MAX:
            // cpu version
            // return *std::max_element(data.begin(), data.end());
            /////////////////////////////////////////////////////
            // gpu version
            return parallel_max(data);
        default:
            throw std::runtime_error("Operation not supported for string columns");
    }
}

template<>
Value AggregateOperator::computeAggregation<bool>(
    const std::vector<bool>& data, parser::AggregateType agg_type) const {
    
    if (data.empty()) {
        return std::monostate{};
    }
    
    switch (agg_type) {
        case parser::AggregateType::COUNT:
            return static_cast<int64_t>(data.size());
        default:
            throw std::runtime_error("Operation not supported for boolean columns");
    }
}

} // namespace execution
} // namespace gpu_dbms