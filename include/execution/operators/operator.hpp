/**
 * @file operator.hpp
 * @brief Base class for query execution operators
 */
#pragma once

#include <memory>
#include <string>
#include "storage/schema.hpp"
#include "execution/result.hpp"

namespace gpu_dbms {
namespace execution {
namespace operators {

/**
 * @enum OperatorType
 * @brief Types of query execution operators
 */
enum class OperatorType {
    SELECT,       // Scan a table
    PROJECTION,   // Project columns
    FILTER,       // Filter rows
    CROSS_PRODUCT, // Cross product of two inputs
    SORT,         // Sort rows
    AGGREGATE,    // Aggregate functions
    SUBQUERY      // Subquery
};

/**
 * @class Operator
 * @brief Base class for query execution operators
 */
class Operator {
public:
    using r = execution::Result;
    
    virtual ~Operator() = default;

    /**
     * @brief Returns the operator type
     * @return The operator type
     */
    virtual OperatorType getType() const = 0;

    /**
     * @brief Returns the output schema of the operator
     * @return The output schema
     */
    virtual std::shared_ptr<storage::Schema> getOutputSchema() const = 0;

    /**
     * @brief Executes the operator and returns the result
     * @return The result of the operation
     */
    virtual std::shared_ptr<r> execute() = 0;

    /**
     * @brief Returns a string representation of the operator for visualization
     * @param indent The indentation level
     * @return The string representation
     */
    virtual std::string toString(int indent = 0) const = 0;

    /**
     * @brief Returns the estimated number of rows this operator will produce
     * @return The estimated row count
     */
    virtual size_t estimateRowCount() const = 0;

    /**
     * @brief Utility function to create an indented string
     * @param indent The indentation level
     * @return A string with the specified indentation
     */
    static std::string getIndentation(int indent) {
        return std::string(indent * 2, ' ');
    }
};

} // namespace operators
} // namespace execution
} // namespace gpu_dbms