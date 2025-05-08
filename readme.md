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
cd ..
```

## Run the project (E2E)

```bash
./gpu_dbms <data_folder_path> <query_file_path>
```

For example

```
.
├── build
├── CMakeLists.txt
├── gpu_dbms
├── SampleTest
│   ├── command.md
│   ├── data
│   │   ├── Employees.csv
│   │   ├── Products.csv
│   │   └── SalesOrders.csv
│   ├── query1.txt
│   ├── query2.txt
│   ├── query3.txt
│   ├── result1.csv
│   ├── result2.csv
│   └── result3.csv

...
```

```bash
./gpu_dbms SampleTest/data SampleTest/query3.txt
```

## Run the Project

After building, run the executable:

```bash
./gpu_dbms --cli
```

This will launch the database shell with a prompt: `gpu-dbms> `

### Load Tables

```bash
.load test_csv_files/employees.csv
.load test_csv_files/departments.csv
.load test_csv_files/projects.csv
```

### Run Queries

```sql
SELECT employee_id, name, salary, age FROM employees ORDER BY age;
```

## Sample Queries
### SELECT

First load table

```bash
.load test_csv_files/employees.csv
```

Then you can try

```sql
SELECT name FROM employees;

SELECT e.name FROM employees as e;

SELECT employee_id, name FROM employees;

SELECT * FROM employees;
```

### WHERE

First load table

```bash
.load test_csv_files/employees.csv
```

Then you can try

```sql
SELECT * FROM employees WHERE salary >  100000;
SELECT * FROM employees WHERE age > 30;
SELECT * FROM employees WHERE salary <= 50000;
SELECT * FROM employees WHERE name = 'Emma Johnson';
SELECT * FROM employees WHERE name != 'Emma Johnson';
```

SELECT AVG(e.Salary) AS AverageSalary FROM Employees e


### AGG

```sql
SELECT sum(employee_id) as sum_1 FROM employees;
```

### ORDER BY

First load table

```bash
.load test_csv_files/employees.csv
```

Then you can try

```sql
SELECT employee_id, name, age FROM employees ORDER BY age;

SELECT employee_id, name, salary, age FROM employees ORDER BY name;

SELECT employee_id, name, salary, age FROM employees ORDER BY salary DESC;

SELECT age, salary FROM employees ORDER BY age ASC, salary DESC;
```

### JOIN

First load tables

```bash
.load test_csv_files/employees.csv
.load test_csv_files/departments.csv
```

Then you can try

```sql
SELECT e.name FROM employees e, departments d WHERE e.department_id = d.department_id;

SELECT e.name, e.salary FROM employees e, departments d WHERE e.department_id = d.department_id AND e.salary > 140000;

SELECT e.name FROM employees e, departments d WHERE e.department_id = d.department_id AND e.salary > 100000;


SELECT e.name, e.salary, e.age FROM employees e, departments d WHERE e.department_id = d.department_id AND e.salary > 100000 AND e.age < 30;

SELECT sum(e.salary) as sum_1 FROM employees e, departments d WHERE e.department_id = d.department_id;
```

### Nested Query

```sql
SELECT employee_id, name FROM (SELECT employee_id, name FROM employees);

SELECT name FROM (SELECT salary, name FROM employees);

SELECT name, department_name FROM (SELECT e.name, e.age, d.department_name FROM employees e, departments d WHERE e.department_id = d.department_id and e.salary > 100000 and e.age > 60);

SELECT avg(salary) FROM (SELECT e.salary, e.age, d.department_name FROM employees e, departments d WHERE e.department_id = d.department_id and e.salary > 100000 and e.age > 60);

SELECT name FROM (SELECT salary, name FROM (SELECT name, age, salary FROM employees));
```