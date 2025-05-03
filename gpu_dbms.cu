// gpu_dbms.c - Main implementation file for GPU-accelerated DBMS
// CSE GPU Architecture & Programming - Spring 2025

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#define MAX_TABLE_NAME 64
#define MAX_COLUMN_NAME 64
#define MAX_COLUMNS 32
#define MAX_STRING_LENGTH 256
#define BLOCK_SIZE 256

// Data types supported by our DBMS
typedef enum {
    INT,
    FLOAT,
    STRING
} DataType;

// Structure to represent a column
typedef struct {
    char name[MAX_COLUMN_NAME];
    DataType type;
    bool isPrimaryKey;
} Column;

// Structure to represent a table schema
typedef struct {
    char name[MAX_TABLE_NAME];
    Column columns[MAX_COLUMNS];
    int columnCount;
    int rowCount;
    int capacity;
} TableSchema;

// Structure to hold data for a single table
typedef struct {
    TableSchema schema;
    void* data;  // Pointer to the actual data (stored in columnar format)
    size_t* offsets;  // Offsets for each column in the data array
} Table;

// Structure to represent a condition
typedef enum {
    EQ,     // Equal
    NEQ,    // Not Equal
    LT,     // Less Than
    GT,     // Greater Than
    LTE,    // Less Than or Equal
    GTE     // Greater Than or Equal
} ConditionType;

typedef struct {
    char columnName[MAX_COLUMN_NAME];
    ConditionType condType;
    DataType valueType;
    union {
        int intValue;
        float floatValue;
        char stringValue[MAX_STRING_LENGTH];
    } value;
} Condition;

// Structure to represent a WHERE clause
typedef struct {
    Condition* conditions;
    int conditionCount;
} WhereClause;

// Structure to represent a join condition
typedef struct {
    char leftTableColumn[MAX_COLUMN_NAME];
    char rightTableColumn[MAX_COLUMN_NAME];
} JoinCondition;

// Structure for query result
typedef struct {
    TableSchema schema;
    void* data;
    size_t* offsets;
    int rowCount;
} QueryResult;

// Utility functions
void initializeTable(Table* table, const char* name);
void addColumn(Table* table, const char* columnName, DataType type, bool isPrimaryKey);
int loadCSVData(Table* table, const char* filename);
void freeTable(Table* table);
void freeQueryResult(QueryResult* result);

// CUDA kernel to perform filter operation based on conditions
__global__ void filterKernel(void* tableData, size_t* columnOffsets, DataType* columnTypes, 
                             int rowCount, Condition* conditions, int conditionCount, 
                             bool* resultMask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < rowCount) {
        resultMask[idx] = true;  // Initially assume this row passes all conditions
        
        // Evaluate each condition
        for (int i = 0; i < conditionCount; i++) {
            Condition cond = conditions[i];
            int columnIdx = 0;  // This would actually be determined based on the column name
            bool conditionResult = false;
            
            // Based on column type
            if (columnTypes[columnIdx] == INT) {
                int* columnData = (int*)(tableData + columnOffsets[columnIdx]);
                int rowValue = columnData[idx];
                
                switch (cond.condType) {
                    case EQ:  conditionResult = (rowValue == cond.value.intValue); break;
                    case NEQ: conditionResult = (rowValue != cond.value.intValue); break;
                    case LT:  conditionResult = (rowValue < cond.value.intValue);  break;
                    case GT:  conditionResult = (rowValue > cond.value.intValue);  break;
                    case LTE: conditionResult = (rowValue <= cond.value.intValue); break;
                    case GTE: conditionResult = (rowValue >= cond.value.intValue); break;
                }
            }
            else if (columnTypes[columnIdx] == FLOAT) {
                float* columnData = (float*)(tableData + columnOffsets[columnIdx]);
                float rowValue = columnData[idx];
                
                switch (cond.condType) {
                    case EQ:  conditionResult = (rowValue == cond.value.floatValue); break;
                    case NEQ: conditionResult = (rowValue != cond.value.floatValue); break;
                    case LT:  conditionResult = (rowValue < cond.value.floatValue);  break;
                    case GT:  conditionResult = (rowValue > cond.value.floatValue);  break;
                    case LTE: conditionResult = (rowValue <= cond.value.floatValue); break;
                    case GTE: conditionResult = (rowValue >= cond.value.floatValue); break;
                }
            }
            // String comparison would be more complex and would need a separate kernel
            
            // Combining conditions with logical AND
            resultMask[idx] = resultMask[idx] && conditionResult;
        }
    }
}

// CUDA kernel to perform hash join
__global__ void buildHashTableKernel(int* keys, int* values, int count, int* hashTable, int tableSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        int key = keys[idx];
        int hashValue = key % tableSize;
        
        // Simple linear probing
        while (true) {
            int oldVal = atomicCAS(&hashTable[hashValue * 2], -1, key);
            if (oldVal == -1 || oldVal == key) {
                hashTable[hashValue * 2 + 1] = values[idx];
                break;
            }
            hashValue = (hashValue + 1) % tableSize;
        }
    }
}

__global__ void probeHashTableKernel(int* keys, int count, int* hashTable, int tableSize, 
                                     int* joinIndices, int* resultCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        int key = keys[idx];
        int hashValue = key % tableSize;
        
        // Simple linear probing to find the key
        bool found = false;
        while (hashTable[hashValue * 2] != -1) {
            if (hashTable[hashValue * 2] == key) {
                int resultIdx = atomicAdd(resultCount, 1);
                joinIndices[resultIdx * 2] = idx;
                joinIndices[resultIdx * 2 + 1] = hashTable[hashValue * 2 + 1];
                found = true;
                break;
            }
            hashValue = (hashValue + 1) % tableSize;
            
            // Break if we've checked all positions
            if (hashValue == key % tableSize && !found) {
                break;
            }
        }
    }
}

// Function to execute a filter operation on GPU
bool* executeFilterGPU(Table* table, WhereClause* whereClause) {
    int rowCount = table->schema.rowCount;
    bool* hostResultMask = (bool*)malloc(rowCount * sizeof(bool));
    bool* deviceResultMask;
    Condition* deviceConditions;
    void* deviceTableData;
    size_t* deviceColumnOffsets;
    DataType* deviceColumnTypes;
    
    // Allocate memory on the device
    cudaMalloc((void**)&deviceResultMask, rowCount * sizeof(bool));
    cudaMalloc((void**)&deviceConditions, whereClause->conditionCount * sizeof(Condition));
    
    // Calculate total size needed for table data
    size_t totalSize = 0;
    DataType* hostColumnTypes = (DataType*)malloc(table->schema.columnCount * sizeof(DataType));
    
    for (int i = 0; i < table->schema.columnCount; i++) {
        hostColumnTypes[i] = table->schema.columns[i].type;
        // Calculate size based on data type
        // This is simplified - you'd need more complex logic for variable-sized types like strings
    }
    
    cudaMalloc((void**)&deviceTableData, totalSize);
    cudaMalloc((void**)&deviceColumnOffsets, table->schema.columnCount * sizeof(size_t));
    cudaMalloc((void**)&deviceColumnTypes, table->schema.columnCount * sizeof(DataType));
    
    // Copy data to device
    cudaMemcpy(deviceConditions, whereClause->conditions, 
               whereClause->conditionCount * sizeof(Condition), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceColumnOffsets, table->offsets, 
               table->schema.columnCount * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceColumnTypes, hostColumnTypes, 
               table->schema.columnCount * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceTableData, table->data, totalSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blocksPerGrid = (rowCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    filterKernel<<<blocksPerGrid, BLOCK_SIZE>>>(deviceTableData, deviceColumnOffsets, 
                                              deviceColumnTypes, rowCount, 
                                              deviceConditions, whereClause->conditionCount, 
                                              deviceResultMask);
    
    // Copy result back to host
    cudaMemcpy(hostResultMask, deviceResultMask, rowCount * sizeof(bool), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(deviceResultMask);
    cudaFree(deviceConditions);
    cudaFree(deviceTableData);
    cudaFree(deviceColumnOffsets);
    cudaFree(deviceColumnTypes);
    
    free(hostColumnTypes);
    
    return hostResultMask;
}

// Function to execute a join operation on GPU using hash join
QueryResult* executeJoinGPU(Table* leftTable, Table* rightTable, JoinCondition* joinCondition) {
    // For this demo, we'll assume we're joining on integer columns
    // In a real implementation, you'd need to handle different data types
    
    // Find the column indices for the join condition
    int leftColIdx = -1, rightColIdx = -1;
    for (int i = 0; i < leftTable->schema.columnCount; i++) {
        if (strcmp(leftTable->schema.columns[i].name, joinCondition->leftTableColumn) == 0) {
            leftColIdx = i;
            break;
        }
    }
    
    for (int i = 0; i < rightTable->schema.columnCount; i++) {
        if (strcmp(rightTable->schema.columns[i].name, joinCondition->rightTableColumn) == 0) {
            rightColIdx = i;
            break;
        }
    }
    
    if (leftColIdx == -1 || rightColIdx == -1) {
        printf("Join columns not found\n");
        return NULL;
    }
    
    // Assuming both join columns are INT type
    int* leftKeys = (int*)(leftTable->data + leftTable->offsets[leftColIdx]);
    int* rightKeys = (int*)(rightTable->data + rightTable->offsets[rightColIdx]);
    
    // Create indices for both tables
    int* leftIndices = (int*)malloc(leftTable->schema.rowCount * sizeof(int));
    int* rightIndices = (int*)malloc(rightTable->schema.rowCount * sizeof(int));
    
    for (int i = 0; i < leftTable->schema.rowCount; i++) {
        leftIndices[i] = i;
    }
    
    for (int i = 0; i < rightTable->schema.rowCount; i++) {
        rightIndices[i] = i;
    }
    
    // Allocate device memory
    int* d_leftKeys, *d_leftIndices;
    int* d_rightKeys, *d_rightIndices;
    
    cudaMalloc((void**)&d_leftKeys, leftTable->schema.rowCount * sizeof(int));
    cudaMalloc((void**)&d_leftIndices, leftTable->schema.rowCount * sizeof(int));
    cudaMalloc((void**)&d_rightKeys, rightTable->schema.rowCount * sizeof(int));
    cudaMalloc((void**)&d_rightIndices, rightTable->schema.rowCount * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_leftKeys, leftKeys, leftTable->schema.rowCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_leftIndices, leftIndices, leftTable->schema.rowCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rightKeys, rightKeys, rightTable->schema.rowCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rightIndices, rightIndices, rightTable->schema.rowCount * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create hash table (twice the size for simple linear probing)
    int hashTableSize = leftTable->schema.rowCount * 2;
    int* hashTable = (int*)malloc(hashTableSize * 2 * sizeof(int));  // *2 for key-value pairs
    memset(hashTable, -1, hashTableSize * 2 * sizeof(int));
    
    int* d_hashTable;
    cudaMalloc((void**)&d_hashTable, hashTableSize * 2 * sizeof(int));
    cudaMemcpy(d_hashTable, hashTable, hashTableSize * 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Build hash table with left table
    int blocksPerGrid = (leftTable->schema.rowCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buildHashTableKernel<<<blocksPerGrid, BLOCK_SIZE>>>(d_leftKeys, d_leftIndices, leftTable->schema.rowCount, d_hashTable, hashTableSize);
    
    // Allocate memory for join results (worst case: cartesian product)
    int maxJoinResults = leftTable->schema.rowCount * rightTable->schema.rowCount;
    int* joinIndices = (int*)malloc(maxJoinResults * 2 * sizeof(int));  // Store pairs of indices
    int* d_joinIndices;
    cudaMalloc((void**)&d_joinIndices, maxJoinResults * 2 * sizeof(int));
    
    // Counter for result size
    int resultCount = 0;
    int* d_resultCount;
    cudaMalloc((void**)&d_resultCount, sizeof(int));
    cudaMemcpy(d_resultCount, &resultCount, sizeof(int), cudaMemcpyHostToDevice);
    
    // Probe hash table with right table
    blocksPerGrid = (rightTable->schema.rowCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    probeHashTableKernel<<<blocksPerGrid, BLOCK_SIZE>>>(d_rightKeys, rightTable->schema.rowCount, d_hashTable, hashTableSize, d_joinIndices, d_resultCount);
    
    // Get result count
    cudaMemcpy(&resultCount, d_resultCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Copy join indices back to host
    cudaMemcpy(joinIndices, d_joinIndices, resultCount * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Create result table schema (combine both tables)
    QueryResult* result = (QueryResult*)malloc(sizeof(QueryResult));
    strcpy(result->schema.name, "JoinResult");
    result->schema.columnCount = leftTable->schema.columnCount + rightTable->schema.columnCount;
    result->rowCount = resultCount;
    
    // Add columns from both tables to result schema
    int colIdx = 0;
    for (int i = 0; i < leftTable->schema.columnCount; i++) {
        result->schema.columns[colIdx] = leftTable->schema.columns[i];
        colIdx++;
    }
    
    for (int i = 0; i < rightTable->schema.columnCount; i++) {
        result->schema.columns[colIdx] = rightTable->schema.columns[i];
        colIdx++;
    }
    
    // Free resources
    free(leftIndices);
    free(rightIndices);
    free(hashTable);
    free(joinIndices);
    
    cudaFree(d_leftKeys);
    cudaFree(d_leftIndices);
    cudaFree(d_rightKeys);
    cudaFree(d_rightIndices);
    cudaFree(d_hashTable);
    cudaFree(d_joinIndices);
    cudaFree(d_resultCount);
    
    // In a real implementation, you would now create the actual joined data
    // by copying the relevant rows from both tables based on the join indices
    
    return result;
}

// Function to execute a projection operation (select specific columns)
QueryResult* executeProjection(Table* table, char** columnNames, int columnCount) {
    // Create result with selected columns
    QueryResult* result = (QueryResult*)malloc(sizeof(QueryResult));
    strcpy(result->schema.name, "ProjectionResult");
    result->schema.columnCount = columnCount;
    result->rowCount = table->schema.rowCount;
    
    // Find column indices in original table
    int* columnIndices = (int*)malloc(columnCount * sizeof(int));
    for (int i = 0; i < columnCount; i++) {
        columnIndices[i] = -1;
        for (int j = 0; j < table->schema.columnCount; j++) {
            if (strcmp(table->schema.columns[j].name, columnNames[i]) == 0) {
                columnIndices[i] = j;
                result->schema.columns[i] = table->schema.columns[j];
                break;
            }
        }
        
        if (columnIndices[i] == -1) {
            printf("Column %s not found in table\n", columnNames[i]);
            free(columnIndices);
            free(result);
            return NULL;
        }
    }
    
    // Allocate memory for result data
    // This is simplified - in real implementation you'd calculate exact sizes needed
    size_t totalSize = table->schema.rowCount * columnCount * sizeof(int);  // Assuming all int for simplicity
    result->data = malloc(totalSize);
    result->offsets = (size_t*)malloc(columnCount * sizeof(size_t));
    
    // Copy selected columns
    for (int i = 0; i < columnCount; i++) {
        result->offsets[i] = i * table->schema.rowCount * sizeof(int);  // Simplified offset calculation
        
        // Copy data from original table
        memcpy((char*)result->data + result->offsets[i], 
               (char*)table->data + table->offsets[columnIndices[i]], 
               table->schema.rowCount * sizeof(int));  // Assuming all int for simplicity
    }
    
    free(columnIndices);
    return result;
}

// Simple CLI for demonstration
void runCLI() {
    printf("GPU-Accelerated DBMS Demo\n");
    printf("Available commands:\n");
    printf("  load <table_name> <csv_file>    - Load data from CSV file\n");
    printf("  query <sql_like_query>          - Execute a query\n");
    printf("  exit                            - Exit the program\n");
    
    char command[1024];
    Table tables[10];  // Support up to 10 tables for demo
    int tableCount = 0;
    
    while (1) {
        printf("> ");
        if (fgets(command, sizeof(command), stdin) == NULL) {
            break;
        }
        
        // Remove newline
        command[strcspn(command, "\n")] = 0;
        
        char cmd[64];
        if (sscanf(command, "%s", cmd) != 1) {
            continue;
        }
        
        if (strcmp(cmd, "exit") == 0) {
            break;
        }
        else if (strcmp(cmd, "load") == 0) {
            char tableName[MAX_TABLE_NAME];
            char fileName[256];
            if (sscanf(command, "load %s %s", tableName, fileName) != 2) {
                printf("Invalid load command\n");
                continue;
            }
            
            initializeTable(&tables[tableCount], tableName);
            // For demo, we'll just add some dummy columns instead of parsing CSV headers
            addColumn(&tables[tableCount], "id", INT, true);
            addColumn(&tables[tableCount], "name", STRING, false);
            addColumn(&tables[tableCount], "age", INT, false);
            
            printf("Table %s loaded with dummy schema\n", tableName);
            tableCount++;
        }
        else if (strcmp(cmd, "query") == 0) {
            // Very simple query parser for demo
            char* queryStart = strstr(command, "query") + 5;
            while (*queryStart == ' ') queryStart++;
            
            printf("Executing query: %s\n", queryStart);
            
            // Mock execution - in real implementation, you'd parse and execute the query
            printf("Query executed successfully\n");
        }
        else {
            printf("Unknown command: %s\n", cmd);
        }
    }
    
    // Cleanup
    for (int i = 0; i < tableCount; i++) {
        freeTable(&tables[i]);
    }
}

int main() {
    printf("GPU-Accelerated DBMS Implementation\n");
    
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        return 1;
    }
    
    // Run the CLI
    runCLI();
    
    // Reset the device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    return 0;
}

// Utility function implementations
void initializeTable(Table* table, const char* name) {
    strcpy(table->schema.name, name);
    table->schema.columnCount = 0;
    table->schema.rowCount = 0;
    table->schema.capacity = 0;
    table->data = NULL;
    table->offsets = NULL;
}

void addColumn(Table* table, const char* columnName, DataType type, bool isPrimaryKey) {
    if (table->schema.columnCount >= MAX_COLUMNS) {
        printf("Max columns reached\n");
        return;
    }
    
    Column* col = &table->schema.columns[table->schema.columnCount];
    strcpy(col->name, columnName);
    col->type = type;
    col->isPrimaryKey = isPrimaryKey;
    
    table->schema.columnCount++;
    
    // Reallocate offsets array
    table->offsets = (size_t*)realloc(table->offsets, table->schema.columnCount * sizeof(size_t));
    
    // Update offsets (simplified - in real implementation, consider data types)
    if (table->schema.columnCount > 1) {
        table->offsets[table->schema.columnCount - 1] = 
            table->offsets[table->schema.columnCount - 2] + 
            (table->schema.capacity * sizeof(int));  // Assuming all int for simplicity
    } else {
        table->offsets[0] = 0;
    }
}

int loadCSVData(Table* table, const char* filename) {
    // Simplified - in real implementation, parse the CSV file
    printf("Loading data from %s\n", filename);
    return 0;  // Success
}

void freeTable(Table* table) {
    if (table->data) {
        free(table->data);
        table->data = NULL;
    }
    
    if (table->offsets) {
        free(table->offsets);
        table->offsets = NULL;
    }
    
    table->schema.columnCount = 0;
    table->schema.rowCount = 0;
    table->schema.capacity = 0;
}

void freeQueryResult(QueryResult* result) {
    if (result->data) {
        free(result->data);
    }
    
    if (result->offsets) {
        free(result->offsets);
    }
    
    free(result);
}