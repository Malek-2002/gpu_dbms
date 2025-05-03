/**
 * @file projection_operator.hpp
 * @brief Projection operator for selecting columns from input
 */
#pragma once

#include "execution/operators/operator.hpp"
#include "parser/query_model.hpp"
#include <vector>

namespace gpu_dbms {
namespace execution {
namespace operators {

/**
 * @class ProjectionOperator
 * @brief Operator that projects columns from its input based on expressions
 */
class ProjectionOperator : public Operator {
public:
    /**
     * @brief Constructs a new ProjectionOperator
     * @param child The input operator
     * @param expressions The expressions to project
     */
    ProjectionOperator(std::shared_ptr<Operator> child, 
                      const std::vector<std::shared_ptr<parser::Expression>>& expressions);

    /**
     * @brief Returns the operator type
     * @return The operator type (PROJECTION)
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
     * @brief Returns the child operator
     * @return The child operator
     */
    std::shared_ptr<Operator> getChild() const { return child_; }

    /**
     * @brief Returns the projection expressions
     * @return The projection expressions
     */
    const std::vector<std::shared_ptr<parser::Expression>>& getExpressions() const { return expressions_; }

private:
    std::shared_ptr<Operator> child_;
    std::vector<std::shared_ptr<parser::Expression>> expressions_;
    std::shared_ptr<storage::Schema> output_schema_;

    /**
     * @brief Builds the output schema based on the expressions
     */
    void buildOutputSchema();
};

} // namespace operators
} // namespace execution
} // namespace gpu_dbms