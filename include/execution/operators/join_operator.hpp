#pragma once

#include "execution/operators/operator.hpp"
#include "parser/sql_parser.hpp"
#include "storage/catalog.hpp"
#include "execution/expression_evaluator.hpp"

namespace gpu_dbms {
namespace execution {

class JoinOperator : public Operator { // Inherit from Operator
public:
    JoinOperator(
        storage::Catalog& catalog,
        const parser::TableRef& left_table,
        const parser::TableRef& right_table,
        std::shared_ptr<parser::Expression> condition,
        std::shared_ptr<storage::Schema> left_schema,
        std::shared_ptr<storage::Schema> right_schema,
        std::shared_ptr<storage::Schema> output_schema
    );

    std::shared_ptr<storage::Schema> getOutputSchema() const override;

    void setInput(std::shared_ptr<Result> input) override;
    void setLeftInput(std::shared_ptr<Result> input);
    void setRightInput(std::shared_ptr<Result> input);

    const parser::TableRef& getLeftTableRef() const;
    const parser::TableRef& getRightTableRef() const;

    std::shared_ptr<Result> execute() override;

private:
    storage::Catalog& catalog_;
    const parser::TableRef& left_table_;
    const parser::TableRef& right_table_;
    std::shared_ptr<parser::Expression> condition_;
    std::shared_ptr<storage::Schema> left_schema_;
    std::shared_ptr<storage::Schema> right_schema_;
    std::shared_ptr<storage::Schema> output_schema_;
    std::shared_ptr<Result> left_input_;
    std::shared_ptr<Result> right_input_;
};

} // namespace execution
} // namespace gpu_dbms