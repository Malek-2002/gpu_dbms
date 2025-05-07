#include "execution/query_executer.hpp"
#include "execution/operators/select_operator.hpp"
#include "execution/operators/filter_operator.hpp"
#include "execution/operators/aggregate_operator.hpp"
#include "execution/operators/join_operator.hpp"
#include "execution/operators/sort_operator.hpp"
#include <stdexcept>
#include <set>
// #include <iostream>

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
    bool is_select_star = false;

    // Check for SELECT * (empty select_list or single non-column expression)
    if (query_model->select_list.empty() || 
        (query_model->select_list.size() == 1 && 
         !std::dynamic_pointer_cast<parser::ColumnExpression>(query_model->select_list[0]) &&
         !std::dynamic_pointer_cast<parser::AggregateExpression>(query_model->select_list[0]))) {
        is_select_star = true;
    }

    // Check for aggregate expressions
    if (!is_select_star && !query_model->select_list.empty()) {
        for (const auto& expr : query_model->select_list) {
            if (auto agg_expr = std::dynamic_pointer_cast<parser::AggregateExpression>(expr)) {
                has_aggregates = true;
                aggregate_exprs.push_back(agg_expr);
            }
        }
    }

    // Build column schemas for output based on select expressions
    if (!is_select_star && !query_model->select_list.empty()) {
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
                throw std::runtime_error("Unsupported expression in select list");
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

    // // Debug: Print output schema
    // std::cerr << "buildQueryPlan: Output schema columns: ";
    // for (const auto& col : output_schema->getColumns()) {
    //     std::cerr << col.name << " ";
    // }
    // std::cerr << std::endl;

    // Now let's build the execution plan
    
    // Step 1: Start with base table operators (one per table)
    std::unordered_map<std::string, std::shared_ptr<Operator>> table_operators;
    std::unordered_map<std::string, std::shared_ptr<storage::Schema>> table_schemas;
    
    for (const auto& table_ref : query_model->tables) {
        std::string table_key = table_ref.alias.empty() ? table_ref.table_name : table_ref.alias;
        auto table_schema = catalog_.getSchema(table_ref.table_name);
        table_schemas[table_key] = output_schema; // Use output_schema for SelectOperator
        // Create a basic select operator for each table
        auto select_op = std::make_shared<SelectOperator>(catalog_, table_ref, output_schema);
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
                        catalog_, table_ref, filter_expr, output_schema);
                    filter_op->setInput(std::shared_ptr<Result>());
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
    
    // // First, initialize with base table results
    // for (const auto& [key, result] : table_results) {
    //     intermediate_results[key] = result;
    // }
    
    // Execute operators in sequence, tracking intermediate results
    for (const auto& op : operators) {
        if (auto select_op = std::dynamic_pointer_cast<SelectOperator>(op)) {
            std::string table_key = select_op->getTableRef().alias.empty() ? 
                select_op->getTableRef().table_name : select_op->getTableRef().alias;
            current_result = table_results[table_key];
            select_op->setInput(current_result);
            current_result = select_op->execute();
        } else if (auto join_op = std::dynamic_pointer_cast<JoinOperator>(op)) {
            std::string left_key = join_op->getLeftTableRef().alias.empty() ? 
                join_op->getLeftTableRef().table_name : join_op->getLeftTableRef().alias;
            std::string right_key = join_op->getRightTableRef().alias.empty() ? 
                join_op->getRightTableRef().table_name : join_op->getRightTableRef().alias;
            
            join_op->setLeftInput(intermediate_results[left_key]);
            join_op->setRightInput(table_results[right_key]);
            current_result = join_op->execute();
            
            intermediate_results["joined_" + std::to_string(intermediate_results.size())] = current_result;
        } else {
            op->setInput(current_result);
            current_result = op->execute();
        }
    }
    
    // // Debug: Print current result schema before projection
    // std::cerr << "executePlan: Current result schema columns: ";
    // for (const auto& col : current_result->getSchema()->getColumns()) {
    //     std::cerr << col.name << " ";
    // }
    // std::cerr << std::endl;

    // Ensure output only includes select_list columns
    if (!query_model->select_list.empty() && 
        !(query_model->select_list.size() == 1 && 
          !std::dynamic_pointer_cast<parser::ColumnExpression>(query_model->select_list[0]) &&
          !std::dynamic_pointer_cast<parser::AggregateExpression>(query_model->select_list[0]))) {
        auto final_schema = std::make_shared<storage::Schema>("final_result");
        for (size_t i = 0; i < query_model->select_list.size(); ++i) {
            std::string col_name;
            if (auto col_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(query_model->select_list[i])) {
                col_name = col_expr->alias.empty() ? col_expr->column.column_name : col_expr->alias;
            } else {
                col_name = "expr_" + std::to_string(i);
            }
            auto col_info = current_result->getSchema()->getColumn(col_name);
            final_schema->addColumn(col_info);
        }

        auto final_result = std::make_shared<Result>(final_schema);
        auto final_table = std::make_shared<storage::Table>(final_schema);
        
        for (const auto& col_info : final_schema->getColumns()) {
            const auto& col_name = col_info.name;
            storage::ColumnData output_col;
            
            switch (col_info.type) {
                case storage::DataType::INT:
                    output_col = storage::IntColumn(current_result->getData()->numRows());
                    break;
                case storage::DataType::FLOAT:
                    output_col = storage::FloatColumn(current_result->getData()->numRows());
                    break;
                case storage::DataType::STRING:
                    output_col = storage::StringColumn(current_result->getData()->numRows());
                    break;
                case storage::DataType::BOOLEAN:
                    output_col = storage::BoolColumn(current_result->getData()->numRows());
                    break;
                default:
                    throw std::runtime_error("Unsupported column type: " + col_name);
            }
            
            final_table->addColumn(col_name, std::move(output_col));
        }
        
        // Copy only selected columns
        ExpressionEvaluator evaluator(current_result->getData());
        for (size_t row = 0; row < current_result->getData()->numRows(); ++row) {
            size_t col_idx = 0;
            for (const auto& expr : query_model->select_list) {
                const auto& col_name = final_schema->getColumns()[col_idx].name;
                auto& result_col = final_table->getColumn(col_name);
                Value val = evaluator.evaluate(expr.get(), row);
                
                switch (final_schema->getColumns()[col_idx].type) {
                    case storage::DataType::INT:
                        std::get<storage::IntColumn>(result_col)[row] = std::holds_alternative<int64_t>(val) ? std::get<int64_t>(val) : 0;
                        break;
                    case storage::DataType::FLOAT:
                        std::get<storage::FloatColumn>(result_col)[row] = std::holds_alternative<double>(val) ? std::get<double>(val) : 0.0;
                        break;
                    case storage::DataType::STRING:
                        std::get<storage::StringColumn>(result_col)[row] = std::holds_alternative<std::string>(val) ? std::get<std::string>(val) : "";
                        break;
                    case storage::DataType::BOOLEAN:
                        std::get<storage::BoolColumn>(result_col)[row] = std::holds_alternative<bool>(val) ? std::get<bool>(val) : false;
                        break;
                    default:
                        throw std::runtime_error("Unsupported column type in final copy");
                }
                ++col_idx;
            }
        }
        
        final_result->setData(final_table);
        current_result = final_result;

        // // Debug: Print final result schema
        // std::cerr << "executePlan: Final result schema columns: ";
        // for (const auto& col : final_result->getSchema()->getColumns()) {
        //     std::cerr << col.name << " ";
        // }
        // std::cerr << std::endl;
    }
    
    return current_result;
}

} // namespace execution
} // namespace gpu_dbms