#pragma once

#include <memory>
#include <vector>
#include "execution/operators/operator.hpp"
#include "parser/query_model.hpp"

namespace gpu_dbms {
namespace execution {

class QueryPlan {
public:
    QueryPlan();
    
    // Add an operator to the plan
    void addOperator(std::shared_ptr<Operator> op);
    
    // Get the list of operators
    const std::vector<std::shared_ptr<Operator>>& getOperators() const;
    
    // Optimize the query plan
    void optimize();
    
private:
    std::vector<std::shared_ptr<Operator>> operators_;
};

} // namespace execution
} // namespace gpu_dbms