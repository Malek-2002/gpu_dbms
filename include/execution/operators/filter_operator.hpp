#pragma once

#include <memory>
#include "execution/operators/operator.hpp"
#include "parser/query_model.hpp"
#include "storage/schema.hpp"
#include "storage/catalog.hpp"

namespace gpu_dbms {
namespace execution {

class FilterOperator : public Operator {
public:
    FilterOperator(storage::Catalog& catalog,
                   const parser::TableRef& table_ref,
                   std::shared_ptr<parser::Expression> filter_expr,
                   std::shared_ptr<storage::Schema> input_schema);

    std::shared_ptr<Result> execute() override;
    void setInput(std::shared_ptr<Result> input) override;
    std::shared_ptr<storage::Schema> getOutputSchema() const override;

private:
    storage::Catalog& catalog_;
    parser::TableRef table_ref_;
    std::shared_ptr<parser::Expression> filter_expr_;
    std::shared_ptr<storage::Schema> input_schema_;
    std::shared_ptr<Result> input_;
};

} // namespace execution
} // namespace gpu_dbms