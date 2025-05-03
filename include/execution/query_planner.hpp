/**
 * @file query_planner.hpp
 * @brief Query planner for generating execution plans from query models
 */
#pragma once

#include <memory>
#include <vector>
#include <string>
#include "parser/query_model.hpp"
#include "execution/query_plan.hpp"
#include "execution/operators/operator.hpp"
#include "storage/catalog.hpp"

namespace gpu_dbms {
namespace execution {

/**
 * @class QueryPlanner
 * @brief Generates execution plans from query models
 */
class QueryPlanner {
public:
    /**
     * @brief Constructs a new QueryPlanner
     * @param catalog The catalog containing schema information
     */
    explicit QueryPlanner(std::shared_ptr<storage::Catalog> catalog);

    /**
     * @brief Generates an execution plan from a query model
     * @param query_model The query model to generate a plan for
     * @return The generated execution plan
     */
    std::shared_ptr<QueryPlan> generatePlan(const parser::QueryModel& query_model);

private:
    std::shared_ptr<storage::Catalog> catalog_;

    /**
     * @brief Generates a logical plan for a query
     * @param query_model The query model to generate a plan for
     * @return The root operator of the logical plan
     */
    std::shared_ptr<operators::Operator> generateLogicalPlan(const parser::QueryModel& query_model);

    /**
     * @brief Generates a scan operator for a table
     * @param table_ref The table reference
     * @return The scan operator
     */
    std::shared_ptr<operators::Operator> generateScanOperator(const parser::TableRef& table_ref);

    /**
     * @brief Generates a filter operator for a where clause
     * @param where_clause The where clause expression
     * @param input The input operator
     * @return The filter operator
     */
    std::shared_ptr<operators::Operator> generateFilterOperator(
        const std::shared_ptr<parser::Expression>& where_clause,
        std::shared_ptr<operators::Operator> input);

    /**
     * @brief Generates a projection operator for select expressions
     * @param select_list The list of select expressions
     * @param input The input operator
     * @return The projection operator
     */
    std::shared_ptr<operators::Operator> generateProjectionOperator(
        const std::vector<std::shared_ptr<parser::Expression>>& select_list,
        std::shared_ptr<operators::Operator> input);

    /**
     * @brief Generates a cross product operator for multiple tables
     * @param left The left input operator
     * @param right The right input operator
     * @return The cross product operator
     */
    std::shared_ptr<operators::Operator> generateCrossProductOperator(
        std::shared_ptr<operators::Operator> left,
        std::shared_ptr<operators::Operator> right);

    /**
     * @brief Generates a sort operator for ORDER BY expressions
     * @param order_by The ORDER BY expressions and sort orders
     * @param input The input operator
     * @return The sort operator
     */
    std::shared_ptr<operators::Operator> generateSortOperator(
        const std::vector<std::pair<std::shared_ptr<parser::Expression>, parser::SortOrder>>& order_by,
        std::shared_ptr<operators::Operator> input);
};

} // namespace execution
} // namespace gpu_dbms