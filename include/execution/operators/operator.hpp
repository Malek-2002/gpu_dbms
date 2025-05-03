#pragma once

#include <memory>
#include "execution/result.hpp"
#include "storage/table.hpp"

namespace gpu_dbms {
namespace execution {

class Operator {
public:
    virtual ~Operator() = default;
    
    // Execute the operator and return result
    virtual std::shared_ptr<Result> execute() = 0;
    
    // Set input data (from previous operator or table)
    virtual void setInput(std::shared_ptr<Result> input) = 0;
    
    // Get output schema
    virtual std::shared_ptr<storage::Schema> getOutputSchema() const = 0;
};

} // namespace execution
} // namespace gpu_dbms