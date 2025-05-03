#pragma once

#include <memory>
#include "execution/operators/operator.hpp"
#include "parser/query_model.hpp"
#include "storage/schema.hpp"

namespace gpu_dbms {
namespace execution {

class JoinOperator : public Operator {
public:
    JoinOperator(parser::JoinType join_type,
                 std::shared_ptr<parser::Expression> join_condition,
                 std::shared_ptr<storage::Schema> left_schema,
                 std::shared_ptr<storage::Schema> right_schema);
    
    std::shared_ptr<Result> execute() override;
    void setInput(std::shared_ptr<Result> input) override;
    void setRightInput(std::shared_ptr<Result> right_input);
    std::shared_ptr<storage::Schema> getOutputSchema() const override;

private:
    parser::JoinType join_type_;
    std::shared_ptr<parser::Expression> join_condition_;
    std::shared_ptr<storage::Schema> left_schema_;
    std::shared_ptr<storage::Schema> right_schema_;
    std::shared_ptr<storage::Schema> output_schema_;
    std::shared_ptr<Result> left_input_;
    std::shared_ptr<Result> right_input_;
};

} // namespace execution
} // namespace gpu_dbms