#include "execution/query_executer.hpp"
#include "execution/operators/select_operator.hpp"
#include "execution/operators/filter_operator.hpp"
#include "execution/operators/aggregate_operator.hpp"
#include "execution/operators/join_operator.hpp"
#include "execution/operators/sort_operator.hpp"
#include <stdexcept>
#include <set>
#include <iostream>

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
                        if (auto col_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(agg_expr->expr)) {
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
                            throw std::runtime_error("Non-column expressions in MIN/MAX not supported");
                        }
                        break;
                    }
                    default:
                        throw std::runtime_error("Unsupported aggregate type");
                }
            } else if (auto col_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(expr)) {
                col_name = col_expr->alias.empty() ? col_expr->column.column_name : col_expr->alias;
                
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
            // Multiple tables: create combined schema without prefixing
            for (const auto& table_ref : query_model->tables) {
                auto table_schema = catalog_.getSchema(table_ref.table_name);
                for (const auto& col_info : table_schema->getColumns()) {
                    storage::ColumnInfo new_col_info{
                        col_info.name, col_info.type, col_info.is_primary_key
                    };
                    output_schema->addColumn(new_col_info);
                }
            }
        }
    }

    // Debug: Print output schema
    std::cerr << "buildQueryPlan: Output schema columns: ";
    for (const auto& col : output_schema->getColumns()) {
        std::cerr << col.name << " ";
    }
    std::cerr << std::endl;

    // Step 1: Start with base table operators (one per table)
    std::unordered_map<std::string, std::shared_ptr<Operator>> table_operators;
    std::unordered_map<std::string, std::shared_ptr<storage::Schema>> table_schemas;
    
    for (const auto& table_ref : query_model->tables) {
        std::string table_key = table_ref.alias.empty() ? table_ref.table_name : table_ref.alias;
        auto table_schema = catalog_.getSchema(table_ref.table_name);
        // Create table-specific schema for SelectOperator
        auto select_schema = std::make_shared<storage::Schema>(table_key + "_select");
        if (!is_select_star && !query_model->select_list.empty()) {
            for (const auto& expr : query_model->select_list) {
                if (auto col_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(expr)) {
                    std::string col_table = col_expr->column.alias.empty() ? col_expr->column.table_name : col_expr->column.alias;
                    // Debug: Trace column and table info
                    std::cerr << "Processing column: " << col_expr->column.column_name 
                              << ", col_table: " << col_table 
                              << ", table_key: " << table_key 
                              << ", table_name: " << table_ref.table_name << std::endl;
                    // Include column if it belongs to this table or table is the only one
                    if (col_table == table_ref.table_name || 
                        col_table == table_key || 
                        (query_model->tables.size() == 1 && col_table.empty())) {
                        auto col_info = table_schema->getColumn(col_expr->column.column_name);
                        if (!select_schema->hasColumn(col_info.name)) {
                            select_schema->addColumn(col_info);
                        }
                    }
                }
            }
        } else {
            // For SELECT *, include all columns from this table
            for (const auto& col_info : table_schema->getColumns()) {
                select_schema->addColumn(col_info);
            }
        }
        // Include join keys for tables involved in joins
        for (const auto& [join_tables, cond_idx] : query_model->join_conditions) {
            for (const auto& table : join_tables) {
                if (table.alias == table_key || table.table_name == table_ref.table_name) {
                    if (auto bin_expr = std::dynamic_pointer_cast<parser::BinaryExpression>(query_model->conditions[cond_idx])) {
                        if (auto col_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(bin_expr->left)) {
                            if (col_expr->column.alias == table_key || col_expr->column.table_name == table_ref.table_name) {
                                auto col_info = table_schema->getColumn(col_expr->column.column_name);
                                if (!select_schema->hasColumn(col_info.name)) {
                                    select_schema->addColumn(col_info);
                                }
                            }
                        }
                        if (auto col_expr = std::dynamic_pointer_cast<parser::ColumnExpression>(bin_expr->right)) {
                            if (col_expr->column.alias == table_key || col_expr->column.table_name == table_ref.table_name) {
                                auto col_info = table_schema->getColumn(col_expr->column.column_name);
                                if (!select_schema->hasColumn(col_info.name)) {
                                    select_schema->addColumn(col_info);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Debug: Print select schema
        std::cerr << "Select schema for " << table_key << ": ";
        for (const auto& col : select_schema->getColumns()) {
            std::cerr << col.name << " ";
        }
        std::cerr << std::endl;

        table_schemas[table_key] = select_schema;
        auto select_op = std::make_shared<SelectOperator>(catalog_, table_ref, select_schema);
        table_operators[table_key] = select_op;

        // Apply table-specific filters only for single-table queries
        if (query_model->tables.size() == 1) {
            auto it = query_model->table_specific_conditions.find(table_key);
            if (it != query_model->table_specific_conditions.end()) {
                for (const auto& cond_idx : it->second) {
                    bool is_join_condition = false;
                    for (const auto& [join_tables, join_idx] : query_model->join_conditions) {
                        if (join_idx == cond_idx) {
                            is_join_condition = true;
                            break;
                        }
                    }
                    
                    if (!is_join_condition) {
                        auto filter_expr = query_model->conditions[cond_idx];
                        std::cerr << "Applying table-specific filter for " << table_key << ": Condition [" << cond_idx << "]" << std::endl;
                        auto filter_op = std::make_shared<FilterOperator>(
                            catalog_, table_ref, filter_expr, select_schema);
                        filter_op->setInput(std::shared_ptr<Result>());
                        table_operators[table_key] = filter_op;
                    } else {
                        std::cerr << "Skipping join condition [" << cond_idx << "] for table-specific filter on " << table_key << std::endl;
                    }
                }
            } else {
                std::cerr << "No table-specific conditions found for " << table_key << std::endl;
            }
        } else {
            std::cerr << "Skipping table-specific filters for " << table_key << " in multi-table query" << std::endl;
        }
    }

    // Step 2: Process joins if there are multiple tables
    std::shared_ptr<Operator> last_operator;
    if (query_model->tables.size() > 1) {
        std::string current_table_key = query_model->tables[0].alias.empty() ? 
            query_model->tables[0].table_name : query_model->tables[0].alias;
        last_operator = table_operators[current_table_key];
        plan->addOperator(last_operator);
        
        auto joined_schema = std::make_shared<storage::Schema>("joined_result");
        for (const auto& col_info : table_schemas[current_table_key]->getColumns()) {
            joined_schema->addColumn(col_info);
        }
        
        for (size_t i = 1; i < query_model->tables.size(); ++i) {
            const auto& right_table_ref = query_model->tables[i];
            std::string right_table_key = right_table_ref.alias.empty() ? 
                right_table_ref.table_name : right_table_ref.alias;
            
            std::shared_ptr<parser::Expression> join_condition = nullptr;
            for (const auto& [join_tables, cond_idx] : query_model->join_conditions) {
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
                    std::cerr << "Using join condition [" << cond_idx << "] for tables " << current_table_key << " and " << right_table_key << std::endl;
                    break;
                }
            }
            
            if (!join_condition) {
                throw std::runtime_error("No join condition found between tables " + 
                                         current_table_key + " and " + right_table_key);
            }
            
            for (const auto& col_info : table_schemas[right_table_key]->getColumns()) {
                joined_schema->addColumn(col_info);
            }
            
            auto join_op = std::make_shared<JoinOperator>(
                catalog_,
                query_model->tables[i-1],
                right_table_ref,
                join_condition,
                table_schemas[current_table_key],
                table_schemas[right_table_key],
                joined_schema
            );
            
            current_table_key = "joined_" + std::to_string(i);
            table_schemas[current_table_key] = joined_schema;
            last_operator = join_op;
            plan->addOperator(last_operator);
        }
    } else {
        std::string table_key = query_model->tables[0].alias.empty() ? 
            query_model->tables[0].table_name : query_model->tables[0].alias;
        last_operator = table_operators[table_key];
        plan->addOperator(last_operator);
    }
    
    // Step 3: Apply global filter conditions (WHERE clause)
    if (query_model->where_clause && !query_model->conditions.empty()) {
        std::vector<size_t> handled_conditions;
        
        // Mark join conditions as handled
        for (const auto& [join_tables, cond_idx] : query_model->join_conditions) {
            handled_conditions.push_back(cond_idx);
            std::cerr << "Marking join condition [" << cond_idx << "] as handled" << std::endl;
        }
        
        // For single-table queries, include table-specific conditions
        if (query_model->tables.size() == 1) {
            std::string table_key = query_model->tables[0].alias.empty() ? 
                query_model->tables[0].table_name : query_model->tables[0].alias;
            auto it = query_model->table_specific_conditions.find(table_key);
            if (it != query_model->table_specific_conditions.end()) {
                for (const auto& idx : it->second) {
                    bool is_join_condition = false;
                    for (const auto& [join_tables, join_idx] : query_model->join_conditions) {
                        if (join_idx == idx) {
                            is_join_condition = true;
                            break;
                        }
                    }
                    if (!is_join_condition) {
                        handled_conditions.push_back(idx);
                        std::cerr << "Marking table-specific condition [" << idx << "] as handled for single-table query" << std::endl;
                    }
                }
            }
        }
        
        std::set<size_t> handled_set(handled_conditions.begin(), handled_conditions.end());
        
        // Collect unhandled conditions
        std::vector<std::shared_ptr<parser::Expression>> unhandled_conditions;
        for (size_t i = 0; i < query_model->conditions.size(); ++i) {
            if (handled_set.find(i) == handled_set.end()) {
                unhandled_conditions.push_back(query_model->conditions[i]);
                std::cerr << "Adding unhandled condition [" << i << "] to global filter" << std::endl;
            }
        }
        
        // If there are unhandled conditions, apply them as a global filter
        if (!unhandled_conditions.empty()) {
            std::shared_ptr<parser::Expression> combined_expr;
            if (unhandled_conditions.size() == 1) {
                combined_expr = unhandled_conditions[0];
            } else {
                combined_expr = unhandled_conditions[0];
                for (size_t i = 1; i < unhandled_conditions.size(); ++i) {
                    auto logical_expr = std::make_shared<parser::LogicalExpression>();
                    logical_expr->op = parser::LogicalOperatorType::AND;
                    logical_expr->left = combined_expr;
                    logical_expr->right = unhandled_conditions[i];
                    combined_expr = logical_expr;
                }
            }
            
            // Debug: Print joined schema before filter
            std::cerr << "Joined schema before global filter: ";
            for (const auto& col : last_operator->getOutputSchema()->getColumns()) {
                std::cerr << col.name << " ";
            }
            std::cerr << std::endl;
            
            auto filter_op = std::make_shared<FilterOperator>(
                catalog_, 
                query_model->tables[0], // Table ref is less relevant post-join
                combined_expr,
                last_operator->getOutputSchema() // Use joined schema
            );
            
            filter_op->setInput(std::shared_ptr<Result>());
            last_operator = filter_op;
            plan->addOperator(last_operator);
            std::cerr << "Applied global filter with schema columns: ";
            for (const auto& col : last_operator->getOutputSchema()->getColumns()) {
                std::cerr << col.name << " ";
            }
            std::cerr << std::endl;
        } else {
            std::cerr << "No unhandled conditions for global filter" << std::endl;
        }
    } else {
        std::cerr << "No WHERE clause or conditions to process" << std::endl;
    }
    
    // Step 4: Apply aggregation if needed
    if (has_aggregates) {
        auto agg_op = std::make_shared<AggregateOperator>(
            catalog_,
            aggregate_exprs,
            last_operator->getOutputSchema(),
            output_schema
        );
        agg_op->setInput(std::shared_ptr<Result>());
        last_operator = agg_op;
        plan->addOperator(last_operator);
    }
    
    // Add SortOperator if ORDER BY clause is present
    if (!query_model->order_by.empty()) {
        auto sort_op = std::make_shared<SortOperator>(
            query_model->order_by,
            last_operator->getOutputSchema()
        );
        last_operator = sort_op;
        plan->addOperator(last_operator);
    }
    
    return plan;
}

std::shared_ptr<Result> QueryExecutor::executePlan(std::shared_ptr<QueryPlan> plan, std::shared_ptr<parser::QueryModel> query_model) {
    if (!plan) {
        throw std::runtime_error("QueryExecutor: Null query plan");
    }
    
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
    
    const auto& operators = plan->getOperators();
    if (operators.empty()) {
        throw std::runtime_error("QueryExecutor: No operators in plan");
    }
    
    std::shared_ptr<Result> current_result;
    std::unordered_map<std::string, std::shared_ptr<Result>> intermediate_results;
    
    for (const auto& op : operators) {
        if (auto select_op = std::dynamic_pointer_cast<SelectOperator>(op)) {
            std::string table_key = select_op->getTableRef().alias.empty() ? 
                select_op->getTableRef().table_name : select_op->getTableRef().alias;
            current_result = table_results[table_key];
            select_op->setInput(current_result);
            current_result = select_op->execute();
            intermediate_results[table_key] = current_result;
            std::cerr << "SelectOperator for " << table_key << ": Output rows: " << current_result->getData()->numRows() << std::endl;
            std::cerr << "SelectOperator schema for " << table_key << ": ";
            for (const auto& col : current_result->getSchema()->getColumns()) {
                std::cerr << col.name << " ";
            }
            std::cerr << std::endl;
        } else if (auto join_op = std::dynamic_pointer_cast<JoinOperator>(op)) {
            std::string left_key = join_op->getLeftTableRef().alias.empty() ? 
                join_op->getLeftTableRef().table_name : join_op->getLeftTableRef().alias;
            std::string right_key = join_op->getRightTableRef().alias.empty() ? 
                join_op->getRightTableRef().table_name : join_op->getRightTableRef().alias;
            
            join_op->setLeftInput(intermediate_results[left_key]);
            join_op->setRightInput(intermediate_results[right_key] ? intermediate_results[right_key] : table_results[right_key]);
            current_result = join_op->execute();
            
            // Debug: Print join result rows and schema
            std::cerr << "JoinOperator: Output rows: " << current_result->getData()->numRows() << std::endl;
            std::cerr << "JoinOperator: Output schema columns: ";
            for (const auto& col : current_result->getSchema()->getColumns()) {
                std::cerr << col.name << " ";
            }
            std::cerr << std::endl;
            
            intermediate_results["joined_" + std::to_string(intermediate_results.size())] = current_result;
        } else if (auto filter_op = std::dynamic_pointer_cast<FilterOperator>(op)) {
            filter_op->setInput(current_result);
            current_result = filter_op->execute();
            std::cerr << "FilterOperator: Output rows: " << current_result->getData()->numRows() << std::endl;
            std::cerr << "FilterOperator: Output schema columns: ";
            for (const auto& col : current_result->getSchema()->getColumns()) {
                std::cerr << col.name << " ";
            }
            std::cerr << std::endl;
        } else {
            op->setInput(current_result);
            current_result = op->execute();
        }
    }
    
    // Debug: Print current result schema before projection
    std::cerr << "executePlan: Current result schema columns: ";
    for (const auto& col : current_result->getSchema()->getColumns()) {
        std::cerr << col.name << " ";
    }
    std::cerr << std::endl;

    // Final projection
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
        
        ExpressionEvaluator evaluator(current_result->getData());
        for (size_t row = 0; row < final_table->numRows(); ++row) {
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

        // Debug: Print final result schema
        std::cerr << "executePlan: Final result schema columns: ";
        for (const auto& col : final_result->getSchema()->getColumns()) {
            std::cerr << col.name << " ";
        }
        std::cerr << std::endl;
    }
    
    return current_result;
}

} // namespace execution
} // namespace gpu_dbms