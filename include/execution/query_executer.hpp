#pragma once

#include <memory>
#include <string>
#include "execution/query_plan.hpp"
#include "execution/result.hpp"
#include "parser/query_model.hpp"
#include "storage/catalog.hpp"

namespace gpu_dbms {
namespace execution {

class QueryExecutor {
public:
    QueryExecutor(storage::Catalog& catalog);
    
    // Execute a parsed query model and return results
    std::shared_ptr<Result> execute(std::shared_ptr<parser::QueryModel> query_model);
    
private:
    // Build query plan from query model
    std::shared_ptr<QueryPlan> buildQueryPlan(std::shared_ptr<parser::QueryModel> query_model);
    
    // Execute the query plan
    std::shared_ptr<Result> executePlan(std::shared_ptr<QueryPlan> plan, std::shared_ptr<parser::QueryModel> query_model);
    
    storage::Catalog& catalog_;
};

} // namespace execution
} // namespace gpu_dbms