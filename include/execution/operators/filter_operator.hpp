/**
 * @file filter_operator.hpp
 * @brief Filter operator for filtering rows based on a condition
 */
#pragma once

#include "execution/operators/operator.hpp"
#include "parser/query_model.hpp"

namespace gpu_dbms {
namespace execution {
namespace operators {

/**
 * @class FilterOperator
 * @brief Operator that filters rows from its input based on a condition
 */
class FilterOperator : public Operator {
public:
    /**
     * @brief Constructs a new FilterOperator
     * @param child The input operator
     * @param condition The filtering condition
     */
    FilterOperator(std::shared_ptr<Operator> child, 
                  std::shared_ptr<parser::Expression> condition);

    /**
     * @brief Returns the operator type
     * @return The operator type (FILTER)
     */
    OperatorType getType() const override;

    /**
     * @brief Returns the output schema of the operator
     * @return The output schema
     */
    std::shared_ptr<storage::Schema> getOutputSchema() const override;

    /**
     * @brief Executes the operator and returns the result
     * @return The result of the operation
     */
    std::shared_ptr<r> execute() override;

    /**
     * @brief Returns a string representation of the operator for visualization
     * @param indent The indentation level
     * @return The string representation
     */
    std::string toString(int indent = 0) const override;

    /**
     * @brief Returns the estimated number of rows this operator will produce
     * @return The estimated row count
     */
    size_t estimateRowCount() const override;

    /**
     * @brief Returns the child operator
     * @return The child operator
     */
    std::shared_ptr<Operator> getChild() const { return child_; }

    /**
     * @brief Returns the filtering condition
     * @return The filtering condition
     */
    std::shared_ptr<parser::Expression> getCondition() const { return condition_; }

private:
    std::shared_ptr<Operator> child_;
    std::shared_ptr<parser::Expression> condition_;
};

} // namespace operators
} // namespace execution
} // namespace gpu_dbms