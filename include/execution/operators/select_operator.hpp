#pragma once

#include <memory>
#include "execution/operators/operator.hpp"
#include "parser/query_model.hpp"
#include "storage/catalog.hpp"

namespace gpu_dbms {
namespace execution {

class SelectOperator : public Operator {
public:
    SelectOperator(storage::Catalog& catalog,
                   const parser::TableRef& table_ref,
                   std::shared_ptr<storage::Schema> output_schema);
    
    std::shared_ptr<Result> execute() override;
    void setInput(std::shared_ptr<Result> input) override;
    std::shared_ptr<storage::Schema> getOutputSchema() const override;
    const parser::TableRef& getTableRef() const; // Added getter for table_ref_

private:
    storage::Catalog& catalog_;
    parser::TableRef table_ref_;
    std::shared_ptr<storage::Schema> output_schema_;
    std::shared_ptr<Result> input_;
};

} // namespace execution
} // namespace gpu_dbms