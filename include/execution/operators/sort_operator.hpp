#pragma once

#include <memory>
#include "execution/operators/operator.hpp"
#include "parser/query_model.hpp"
#include "storage/schema.hpp"

namespace gpu_dbms {
namespace execution {

class SortOperator : public Operator {
public:
    SortOperator(const std::vector<std::pair<std::shared_ptr<parser::Expression>, parser::SortOrder>>& sort_keys,
                 std::shared_ptr<storage::Schema> input_schema);
    
    std::shared_ptr<Result> execute() override;
    void setInput(std::shared_ptr<Result> input) override;
    std::shared_ptr<storage::Schema> getOutputSchema() const override;

private:
    std::vector<std::pair<std::shared_ptr<parser::Expression>, parser::SortOrder>> sort_keys_;
    std::shared_ptr<storage::Schema> input_schema_;
    std::shared_ptr<Result> input_;
};

} // namespace execution
} // namespace gpu_dbms