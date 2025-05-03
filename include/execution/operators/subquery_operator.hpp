#pragma once

#include <memory>
#include "execution/operators/operator.hpp"
#include "execution/query_plan.hpp"
#include "storage/schema.hpp"

namespace gpu_dbms {
namespace execution {

class SubqueryOperator : public Operator {
public:
    SubqueryOperator(std::shared_ptr<QueryPlan> subquery_plan);

    std::shared_ptr<Result> execute() override;
    void setInput(std::shared_ptr<Result> input) override;
    std::shared_ptr<storage::Schema> getOutputSchema() const override;

private:
    std::shared_ptr<QueryPlan> subquery_plan_;
    std::shared_ptr<storage::Schema> output_schema_;
    std::shared_ptr<Result> input_;
};

} // namespace execution
} // namespace gpu_dbms