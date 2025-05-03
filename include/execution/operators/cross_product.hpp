/**
 * @file cross_product_operator.hpp
 * @brief Cross product operator for combining two input operators
 */
#pragma once

#include "execution/operators/operator.hpp"

namespace gpu_dbms {
namespace execution {
namespace operators {

/**
 * @class CrossProductOperator
 * @brief Operator that performs a cross product of two input operators
 */
class CrossProductOperator : public Operator {
public:
    /**
     * @brief Constructs a new CrossProductOperator
     * @param left_child The left input operator
     * @param right_child The right input operator
     */
    CrossProductOperator(std::shared_ptr<Operator> left_child, std::shared_ptr<Operator> right_child);

    /**
     * @brief Returns the operator type
     * @return The operator type (CROSS_PRODUCT)
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
    std::shared_ptr<Result> execute() override;

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
     * @brief Returns the left child operator
     * @return The left child operator
     */
    std::shared_ptr<Operator> getLeftChild() const { return left_child_; }

    /**
     * @brief Returns the right child operator
     * @return The right child operator
     */
    std::shared_ptr<Operator> getRightChild() const { return right_child_; }

private:
    std::shared_ptr<Operator> left_child_;
    std::shared_ptr<Operator> right_child_;
    std::shared_ptr<storage::Schema> output_schema_;
};

} // namespace operators
} // namespace execution
} // namespace gpu_dbms