#include "execution/query_plan.hpp"
#include <stdexcept>

namespace gpu_dbms {
namespace execution {

QueryPlan::QueryPlan() : operators_() {}

void QueryPlan::addOperator(std::shared_ptr<Operator> op) {
    if (!op) {
        throw std::runtime_error("QueryPlan: Cannot add null operator");
    }
    operators_.push_back(op);
}

const std::vector<std::shared_ptr<Operator>>& QueryPlan::getOperators() const {
    return operators_;
}

void QueryPlan::optimize() {
    // TODO: Implement query plan optimization
    // Possible optimizations:
    // - Reorder operators (e.g., push filters before projections)
    // - Merge compatible operators
    // - Apply predicate pushdown
    // For now, this is a no-op
}

} // namespace execution
} // namespace gpu_dbms