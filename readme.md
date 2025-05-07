## Prerequisites

Install the required tools and libraries:

```bash
sudo apt update
sudo apt install build-essential cmake bison flex
```

## External Dependency: SQL Parser

Clone and build the external SQL parser:

### Download

```bash
git submodule update --init --recursive
```

### Build

```bash
cd extern/sql-parser
make
cd ../..
```

## Build the Project

Create a build directory and configure the project:

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Run the Project

After building, run the executable:

```bash
./gpu_dbms
```

This will launch the database shell with a prompt: `gpu-dbms> `

### Load Tables

```bash
.load ../test_csv_files/employees.csv employees
.load ../test_csv_files/departments.csv departments
.load ../test_csv_files/projects.csv projects
```

### Run Queries

```sql
SELECT employee_id, name, salary, age FROM employees ORDER BY age;
```

## Sample Queries
### SELECT

First load table

```bash
.load ../test_csv_files/employees.csv employees
```

Then you can try

```sql
SELECT name FROM employees;

SELECT employee_id, name FROM employees;

SELECT * FROM employees;
```

### ORDER BY

First load table

```bash
.load ../test_csv_files/employees.csv employees
```

Then you can try

```sql
SELECT employee_id, name, salary, age FROM employees ORDER BY age;

SELECT employee_id, name, salary, age FROM employees ORDER BY name;

SELECT employee_id, name, salary, age FROM employees ORDER BY salary DESC;

SELECT employee_id, name, salary, age FROM employees ORDER BY age ASC, salary DESC;
```