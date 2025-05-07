#include "execution/query_executer.hpp"
#include "execution/operators/select_operator.hpp"
#include "execution/operators/filter_operator.hpp"
#include "execution/operators/aggregate_operator.hpp"
#include "execution/operators/join_operator.hpp"
#include "execution/operators/sort_operator.hpp"
#include <stdexcept>
#include <set>

namespace gpu_dbms {
namespace execution {

// Enhanced implementation of the QueryExecutor class

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

    // Create output schema for the final operator
    auto output_schema = std::make_shared<storage::Schema>("query_result");
    std::vector<std::shared_ptr<parser::Expression>> aggregate_exprs;
    bool has_aggregates = false;

    // Check for aggregate expressions
    if (!query_model->select_list.empty()) {
        for (const auto& expr : query_model->select_list) {
            if (auto agg_expr = std::dynamic_pointer_cast<parser::AggregateExpression>(expr)) {
                has_aggregates = true;
                aggregate_exprs.push_back(agg_expr);
            }
        }
    }

    // Build column schemas for output based on select expressions
    if (!query_model->select_list.empty()) {
        for (size_t i = 0; i < query_model->select_list.size(); ++i) {
            const auto& expr = query_model->select_list[i];
            std::string col_name;
            storage::DataType col_type = storage::DataType::INT; // Default, to be determined

            if (auto agg_expr = std::dynamic_pointer_cast<parser::AggregateExpression>(expr)) {
                col_name = agg_expr->alias.empty() ? "agg_" + std::to_string(i) : agg_expr->alias;

                // Determine output type based on aggregate type
                switch (agg_expr->type) {
                    case parser::AggregateType::COUNT:
                        col_type = storage::DataType::INT;
                        break;
                    case parser::AggregateType::SUM:
                    case parser::AggregateType::AVG:
                        col_type = storage::DataType::FLOAT;
                        break;
                    case parser::AggregateType::MIN:
                    case parser::AggregateType::MAX: {
                        // Get type from the underlying column expression
                        if (auto col_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(agg_expr->expr)) {
                            std::string table_name;
                            // Find the real table name from the query model
                            for (const auto& table : query_model->tables) {
                                if (table.alias == col_expr->column.alias || 
                                    (col_expr->column.alias.empty() && table.table_name == col_expr->column.table_name)) {
                                    table_name = table.table_name;
                                    break;
                                }
                            }
                            if (table_name.empty()) {
                                table_name = col_expr->column.table_name;
                            }
                            auto table_schema = catalog_.getSchema(table_name);
                            auto col_info = table_schema->getColumn(col_expr->column.column_name);
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
                
                // Find the real table name from the query model
                std::string table_name;
                for (const auto& table : query_model->tables) {
                    if (table.alias == col_expr->column.alias || 
                        (col_expr->column.alias.empty() && table.table_name == col_expr->column.table_name)) {
                        table_name = table.table_name;
                        break;
                    }
                }
                if (table_name.empty()) {
                    table_name = col_expr->column.table_name;
                }
                
                auto table_schema = catalog_.getSchema(table_name);
                auto col_info = table_schema->getColumn(col_expr->column.column_name);
                col_type = col_info.type;
            } else {
                col_name = "expr_" + std::to_string(i);
                col_type = storage::DataType::INT; // Fallback for non-column expressions
            }

            // Fix: Create ColumnInfo with only the members it actually has
            storage::ColumnInfo col_info{col_name, col_type, false};
            output_schema->addColumn(col_info);
        }
    } else {
        // For implicit SELECT *
        if (query_model->tables.size() == 1) {
            // Single table: use its schema directly
            auto table_schema = catalog_.getSchema(query_model->tables[0].table_name);
            for (const auto& col_info : table_schema->getColumns()) {
                output_schema->addColumn(col_info);
            }
        } else {
            // Multiple tables: create combined schema
            for (const auto& table_ref : query_model->tables) {
                auto table_schema = catalog_.getSchema(table_ref.table_name);
                for (const auto& col_info : table_schema->getColumns()) {
                    // Add prefix to avoid column name conflicts
                    std::string prefixed_name = table_ref.alias.empty() ? 
                        table_ref.table_name + "." + col_info.name : 
                        table_ref.alias + "." + col_info.name;
                    
                    // Fix: Create ColumnInfo with only the members it actually has
                    storage::ColumnInfo new_col_info{
                        prefixed_name, col_info.type, col_info.is_primary_key
                    };
                    output_schema->addColumn(new_col_info);
                }
            }
        }
    }

    // Now let's build the execution plan
    
    // Step 1: Start with base table operators (one per table)
    std::unordered_map<std::string, std::shared_ptr<Operator>> table_operators;
    std::unordered_map<std::string, std::shared_ptr<storage::Schema>> table_schemas;
    
    for (const auto& table_ref : query_model->tables) {
        std::string table_key = table_ref.alias.empty() ? table_ref.table_name : table_ref.alias;
        auto table_schema = catalog_.getSchema(table_ref.table_name);
        table_schemas[table_key] = table_schema;
        
        // Create a basic select operator for each table
        auto select_op = std::make_shared<SelectOperator>(catalog_, table_ref, table_schema);
        table_operators[table_key] = select_op;
        
        // Apply table-specific filters if any
        auto it = query_model->table_specific_conditions.find(table_key);
        if (it != query_model->table_specific_conditions.end()) {
            // Only apply filters that are specific to this table and not join conditions
            for (const auto& cond_idx : it->second) {
                // Check if this is not a join condition (i.e., only references this table)
                bool is_join_condition = false;
                for (const auto& [join_tables, join_idx] : query_model->join_conditions) {
                    if (join_idx == cond_idx) {
                        is_join_condition = true;
                        break;
                    }
                }
                
                if (!is_join_condition) {
                    auto filter_expr = query_model->conditions[cond_idx];
                    auto filter_op = std::make_shared<FilterOperator>(
                        catalog_, table_ref, filter_expr, table_schema);
                    filter_op->setInput(std::shared_ptr<Result>()); // Will be set during execution
                    table_operators[table_key] = filter_op;
                }
            }
        }
    }

    // Step 2: Process joins if there are multiple tables
    std::shared_ptr<Operator> last_operator;
    if (query_model->tables.size() > 1) {
        // We need to build join operators
        // Start with the first table
        std::string current_table_key = query_model->tables[0].alias.empty() ? 
            query_model->tables[0].table_name : query_model->tables[0].alias;
        last_operator = table_operators[current_table_key];
        
        // Create schema for the joined result
        auto joined_schema = std::make_shared<storage::Schema>("joined_result");
        
        // Add columns from the first table
        for (const auto& col_info : table_schemas[current_table_key]->getColumns()) {
            joined_schema->addColumn(col_info);
        }
        
        // For each additional table, create a join operator
        for (size_t i = 1; i < query_model->tables.size(); ++i) {
            const auto& right_table_ref = query_model->tables[i];
            std::string right_table_key = right_table_ref.alias.empty() ? 
                right_table_ref.table_name : right_table_ref.alias;
            
            // Find join condition for these tables
            std::shared_ptr<parser::Expression> join_condition = nullptr;
            for (const auto& [join_tables, cond_idx] : query_model->join_conditions) {
                // Check if this join condition involves both tables
                bool involves_both_tables = false;
                bool involves_current_table = false;
                bool involves_right_table = false;
                
                for (const auto& table : join_tables) {
                    std::string table_key = table.alias.empty() ? table.table_name : table.alias;
                    if (table_key == current_table_key) {
                        involves_current_table = true;
                    }
                    if (table_key == right_table_key) {
                        involves_right_table = true;
                    }
                }
                
                involves_both_tables = involves_current_table && involves_right_table;
                if (involves_both_tables) {
                    join_condition = query_model->conditions[cond_idx];
                    break;
                }
            }
            
            if (!join_condition) {
                // If no explicit join condition was found, use a cross join
                // or throw an error if explicit join conditions are required
                throw std::runtime_error("No join condition found between tables " + 
                                         current_table_key + " and " + right_table_key);
            }
            
            // Add columns from the right table to the joined schema
            for (const auto& col_info : table_schemas[right_table_key]->getColumns()) {
                joined_schema->addColumn(col_info);
            }
            
            // Create join operator
            auto join_op = std::make_shared<JoinOperator>(
                catalog_,
                query_model->tables[i-1],  // Left table
                right_table_ref,          // Right table
                join_condition,
                table_schemas[current_table_key],
                table_schemas[right_table_key],
                joined_schema
            );
            
            // Update the current table for the next iteration
            current_table_key = "joined_" + std::to_string(i);
            table_schemas[current_table_key] = joined_schema;
            last_operator = join_op;
        }
    } else {
        // Single table query - use the table operator directly
        std::string table_key = query_model->tables[0].alias.empty() ? 
            query_model->tables[0].table_name : query_model->tables[0].alias;
        last_operator = table_operators[table_key];
    }
    
    // Step 3: Apply global filter conditions (WHERE clause)
    if (query_model->where_clause && !query_model->conditions.empty()) {
        // Create a global filter for any conditions not handled by table-specific filters or joins
        std::vector<size_t> handled_conditions;
        
        // Collect all condition indices that have been handled
        for (const auto& [table_key, cond_indices] : query_model->table_specific_conditions) {
            for (const auto& idx : cond_indices) {
                handled_conditions.push_back(idx);
            }
        }
        
        for (const auto& [join_tables, cond_idx] : query_model->join_conditions) {
            handled_conditions.push_back(cond_idx);
        }
        
        // Create a set for faster lookup
        std::set<size_t> handled_set(handled_conditions.begin(), handled_conditions.end());
        
        // Find unhandled conditions
        std::vector<std::shared_ptr<parser::Expression>> unhandled_conditions;
        for (size_t i = 0; i < query_model->conditions.size(); ++i) {
            if (handled_set.find(i) == handled_set.end()) {
                unhandled_conditions.push_back(query_model->conditions[i]);
            }
        }
        
        // If there are unhandled conditions, create a global filter
        if (!unhandled_conditions.empty()) {
            // Combine unhandled conditions with AND
            std::shared_ptr<parser::Expression> combined_expr = unhandled_conditions[0];
            for (size_t i = 1; i < unhandled_conditions.size(); ++i) {
                auto logical_expr = std::make_shared<parser::LogicalExpression>();
                logical_expr->op = parser::LogicalOperatorType::AND;
                logical_expr->left = combined_expr;
                logical_expr->right = unhandled_conditions[i];
                combined_expr = logical_expr;
            }
            
            // Create the global filter operator
            auto filter_op = std::make_shared<FilterOperator>(
                catalog_, 
                query_model->tables[0],  // Use first table for reference (not important here)
                combined_expr,
                last_operator->getOutputSchema()
            );
            
            filter_op->setInput(std::shared_ptr<Result>()); // Will be set during execution
            last_operator = filter_op;
        }
    }
    
    // Step 4: Apply aggregation if needed
    if (has_aggregates) {
        auto agg_op = std::make_shared<AggregateOperator>(
            catalog_,
            aggregate_exprs,
            last_operator->getOutputSchema(),
            output_schema
        );
        agg_op->setInput(std::shared_ptr<Result>());  // Will be set during execution
        last_operator = agg_op;
    }
    
    // Add SortOperator if ORDER BY clause is present
    if (!query_model->order_by.empty()) {
        auto sort_op = std::make_shared<SortOperator>(
            query_model->order_by,
            last_operator->getOutputSchema()
        );
        sort_op->setInput(std::shared_ptr<Result>());
        last_operator = sort_op;
    }
    
    // Add the final operator to the plan
    plan->addOperator(last_operator);
    
    return plan;
}

std::shared_ptr<Result> QueryExecutor::executePlan(std::shared_ptr<QueryPlan> plan, std::shared_ptr<parser::QueryModel> query_model) {
    if (!plan) {
        throw std::runtime_error("QueryExecutor: Null query plan");
    }
    
    // Initialize result map for each table
    std::unordered_map<std::string, std::shared_ptr<Result>> table_results;
    for (const auto& table_ref : query_model->tables) {
        std::string table_key = table_ref.alias.empty() ? table_ref.table_name : table_ref.alias;
        auto table = catalog_.getTable(table_ref.table_name);
        if (!table) {
            throw std::runtime_error("QueryExecutor: Table '" + table_ref.table_name + "' not found in catalog");
        }
        auto schema = catalog_.getSchema(table_ref.table_name);
        auto result = std::make_shared<Result>(schema);
        result->setData(table);
        table_results[table_key] = result;
    }
    
    // Get all operators from the plan
    const auto& operators = plan->getOperators();
    if (operators.empty()) {
        throw std::runtime_error("QueryExecutor: No operators in plan");
    }
    
    // Improved execution logic for handling joins
    std::shared_ptr<Result> current_result;
    std::unordered_map<std::string, std::shared_ptr<Result>> intermediate_results;
    
    // First, initialize with base table results
    for (const auto& [key, result] : table_results) {
        intermediate_results[key] = result;
    }
    
    // Execute operators in sequence, tracking intermediate results
    for (const auto& op : operators) {
        if (auto join_op = std::dynamic_pointer_cast<JoinOperator>(op)) {
            // For join operators, identify and set both inputs properly
            const auto& left_table_ref = join_op->getLeftTableRef();
            const auto& right_table_ref = join_op->getRightTableRef();
            
            std::string left_key = left_table_ref.alias.empty() ? 
                left_table_ref.table_name : left_table_ref.alias;
                
            std::string right_key = right_table_ref.alias.empty() ? 
                right_table_ref.table_name : right_table_ref.alias;
            
            // Get the appropriate results for left and right inputs
            auto left_result = intermediate_results[left_key];
            auto right_result = intermediate_results[right_key];
            
            if (!left_result || !right_result) {
                throw std::runtime_error("QueryExecutor: Missing input for join operation");
            }
            
            // Set inputs and execute join
            join_op->setLeftInput(left_result);
            join_op->setRightInput(right_result);
            current_result = join_op->execute();
            
            // Store the joined result with a unique key
            std::string join_key = "joined_" + left_key + "_" + right_key;
            intermediate_results[join_key] = current_result;
            
            // Update both tables' entries to point to this joined result
            intermediate_results[left_key] = current_result;
            intermediate_results[right_key] = current_result;
        } else {
            // For non-join operators, set input from previous result
            if (!current_result) {
                // If this is the first operator, use table data
                if (query_model->tables.size() == 1) {
                    std::string table_key = query_model->tables[0].alias.empty() ? 
                        query_model->tables[0].table_name : query_model->tables[0].alias;
                    op->setInput(table_results[table_key]);
                } else {
                    // For multiple tables, we should have processed a join first
                    throw std::runtime_error("QueryExecutor: Expected prior join result for multiple tables");
                }
            } else {
                op->setInput(current_result);
            }
            
            current_result = op->execute();
        }
    }
    
    return current_result;
}
} // namespace execution
} // namespace gpu_dbms