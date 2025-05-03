/**
 * @file query_plan.hpp
 * @brief Execution plan for a query
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "execution/operators/operator.hpp"

namespace gpu_dbms {
namespace execution {

/**
 * @class QueryPlan
 * @brief Represents an execution plan for a query
 */
class QueryPlan {
public:
    /**
     * @brief Constructs a new QueryPlan
     * @param root_operator The root operator of the plan
     */
    explicit QueryPlan(std::shared_ptr<operators::Operator> root_operator);

    /**
     * @brief Returns the root operator of the plan
     * @return The root operator
     */
    std::shared_ptr<operators::Operator> getRootOperator() const;

    /**
     * @brief Executes the query plan and returns the result
     * @return The result of executing the plan
     */
    std::shared_ptr<operators::Operator::r> execute();

    /**
     * @brief Returns a string representation of the query plan
     * @return The string representation
     */
    std::string toString() const;

    /**
     * @brief Returns a visual representation of the query plan as a tree
     * @return The visual representation
     */
    std::string visualizeTree() const;

private:
    std::shared_ptr<operators::Operator> root_operator_;

    /**
     * @brief Helper method to visualize the operator tree
     * @param op The current operator
     * @param indent The indentation level
     * @param is_last Whether this is the last child of its parent
     * @param prefix The prefix for the current line
     * @param result The resulting string
     */
    void visualizeTreeHelper(
        const std::shared_ptr<operators::Operator>& op,
        int indent,
        bool is_last,
        const std::string& prefix,
        std::string& result) const;

    /**
     * @brief Recursively gets all operators in the plan
     * @param op The current operator
     * @param operators The resulting vector of operators
     */
    void getAllOperators(
        const std::shared_ptr<operators::Operator>& op,
        std::vector<std::shared_ptr<operators::Operator>>& operators) const;
};

} // namespace execution
} // namespace gpu_dbms