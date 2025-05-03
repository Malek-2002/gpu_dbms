#include "execution/query_executer.hpp"
#include "execution/operators/select_operator.hpp"
#include "execution/operators/filter_operator.hpp"
#include "execution/operators/aggregate_operator.hpp"
#include <stdexcept>

namespace gpu_dbms {
namespace execution {

QueryExecutor::QueryExecutor(storage::Catalog& catalog) : catalog_(catalog) {}

std::shared_ptr<Result> QueryExecutor::execute(std::shared_ptr<parser::QueryModel> query_model) {
    if (!query_model) {
        throw std::runtime_error("QueryExecutor: Null query model");
    }

    // Build and execute the query plan
    auto plan = buildQueryPlan(query_model);
    return executePlan(plan, query_model);
}

std::shared_ptr<QueryPlan> QueryExecutor::buildQueryPlan(std::shared_ptr<parser::QueryModel> query_model) {
    auto plan = std::make_shared<QueryPlan>();

    // Handle subqueries (if present)
    if (query_model->hasSubquery()) {
        throw std::runtime_error("QueryExecutor: Subqueries not yet supported");
    }

    // Validate table references
    if (query_model->tables.empty()) {
        throw std::runtime_error("QueryExecutor: No tables specified in query");
    }
    if (query_model->tables.size() > 1) {
        throw std::runtime_error("QueryExecutor: Multiple table joins not yet supported");
    }

    // Get table information
    const auto& table_ref = query_model->tables[0];
    const auto& table_name = table_ref.table_name;
    if (!catalog_.hasTable(table_name)) {
        throw std::runtime_error("QueryExecutor: Table '" + table_name + "' not found in catalog");
    }
    auto input_schema = catalog_.getSchema(table_name);

    // Create output schema for the final operator
    auto output_schema = std::make_shared<storage::Schema>("query_result");
    std::vector<std::shared_ptr<parser::Expression>> aggregate_exprs;
    bool has_aggregates = false;

    // Check for aggregate expressions and build output schema
    if (!query_model->select_list.empty()) {
        for (size_t i = 0; i < query_model->select_list.size(); ++i) {
            const auto& expr = query_model->select_list[i];
            std::string col_name;
            storage::DataType col_type = storage::DataType::INT; // Default, to be determined

            if (auto agg_expr = std::dynamic_pointer_cast<parser::AggregateExpression>(expr)) {
                has_aggregates = true;
                aggregate_exprs.push_back(agg_expr);
                col_name = agg_expr->alias.empty() ? "agg_" + std::to_string(i) : agg_expr->alias;

                // Determine output type based on aggregate type
                switch (agg_expr->type) {
                    case parser::AggregateType::COUNT:
                        col_type = storage::DataType::INT;
                        break;
                    case parser::AggregateType::SUM:
                    case parser::AggregateType::AVG:
                        col_type = storage::DataType::FLOAT; // SUM and AVG may return floating-point
                        break;
                    case parser::AggregateType::MIN:
                    case parser::AggregateType::MAX: {
                        // Get type from the underlying column expression
                        if (auto col_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(agg_expr->expr)) {
                            auto col_info = input_schema->getColumn(col_expr->column.column_name);
                            col_type = col_info.type;
                        } else {
                            throw std::runtime_error("Non-column expressions in MIN/MAX not supported");
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error("Unsupported aggregate type");
                }
            } else if (auto col_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(expr)) {
                col_name = col_expr->alias.empty() ? col_expr->column.column_name : col_expr->alias;
                auto col_info = input_schema->getColumn(col_expr->column.column_name);
                col_type = col_info.type;
            } else {
                col_name = "expr_" + std::to_string(i);
                col_type = storage::DataType::INT; // Fallback for non-column expressions
            }

            storage::ColumnInfo col_info{col_name, col_type, false, false, "", ""};
            output_schema->addColumn(col_info);
        }
    } else {
        // Implicit SELECT *: Use all columns from input schema
        for (const auto& col_info : input_schema->getColumns()) {
            output_schema->addColumn(col_info);
        }
    }

    // Add FilterOperator for WHERE clause (if present)
    if (query_model->where_clause) {
        auto filter_op = std::make_shared<FilterOperator>(catalog_, table_ref, query_model->where_clause, input_schema);
        plan->addOperator(filter_op);
    }

    // Add AggregateOperator if there are aggregate expressions
    if (has_aggregates) {
        auto agg_op = std::make_shared<AggregateOperator>(catalog_, aggregate_exprs, input_schema, output_schema);
        plan->addOperator(agg_op);
    } else {
        // Add SelectOperator for non-aggregate queries
        auto select_op = std::make_shared<SelectOperator>(catalog_, table_ref, output_schema);
        plan->addOperator(select_op);
    }

    // Optimize the plan (placeholder)
    // plan->optimize();

    return plan;
}

std::shared_ptr<Result> QueryExecutor::executePlan(std::shared_ptr<QueryPlan> plan, std::shared_ptr<parser::QueryModel> query_model) {
    if (!plan) {
        throw std::runtime_error("QueryExecutor: Null query plan");
    }   
    const auto& table_ref = query_model->tables[0];

    // Initialize current_result with table data from catalog
    std::shared_ptr<Result> current_result;
    if (!table_ref.table_name.empty()) {
        auto table = catalog_.getTable(table_ref.table_name);
        if (!table) {
            throw std::runtime_error("QueryExecutor: Table '" + table_ref.table_name + "' not found in catalog");
        }
        auto schema = catalog_.getSchema(table_ref.table_name);
        current_result = std::make_shared<Result>(schema);
        current_result->setData(table);
    } else {
        throw std::runtime_error("QueryExecutor: Invalid table reference");
    }

    // Execute each operator in sequence
    const auto& operators = plan->getOperators();
    for (size_t i = 0; i < operators.size(); ++i) {
        const auto& op = operators[i];
        // Set input for all operators
        if (!current_result) {
            throw std::runtime_error("QueryExecutor: Null intermediate result");
        }
        op->setInput(current_result);
        current_result = op->execute();
    }

    return current_result;
}

} // namespace execution
} // namespace gpu_dbms