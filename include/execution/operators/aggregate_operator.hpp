#pragma once

#include <memory>
#include <string>
#include <vector>
#include "execution/operators/operator.hpp"
#include "execution/result.hpp"
#include "parser/query_model.hpp"
#include "execution/expression_evaluator.hpp"
#include "storage/catalog.hpp"
#include "storage/schema.hpp"

namespace gpu_dbms {
namespace execution {

class AggregateOperator : public Operator {
public:
AggregateOperator(storage::Catalog& catalog,
    std::vector<std::shared_ptr<parser::Expression>> exprs,
    std::shared_ptr<storage::Schema> input_schema,
    std::shared_ptr<storage::Schema> output_schema);

    // Execute the aggregate operator
    std::shared_ptr<Result> execute() override;

    // Set the input result
    void setInput(std::shared_ptr<Result> input) override;

    // Get the output schema
    std::shared_ptr<storage::Schema> getOutputSchema() const override;

    std::string getAggregateTypeName(parser::AggregateType type) const;
private:
    // Helper method to compute aggregations
    template<typename T>
    Value computeAggregation(const std::vector<T>& data, parser::AggregateType agg_type) const;

    // Perform specific aggregations
    Value computeCount(const storage::ColumnData& col_data) const;
    Value computeSum(const storage::ColumnData& col_data) const;
    Value computeAvg(const storage::ColumnData& col_data) const;
    Value computeMin(const storage::ColumnData& col_data) const;
    Value computeMax(const storage::ColumnData& col_data) const;

    storage::Catalog& catalog_;
    parser::TableRef table_ref_;
    std::vector<std::shared_ptr<parser::AggregateExpression>> agg_exprs_;
    std::shared_ptr<storage::Schema> input_schema_;
    std::shared_ptr<storage::Schema> output_schema_;
    std::shared_ptr<Result> input_;
};

} // namespace execution
} // namespace gpu_dbms