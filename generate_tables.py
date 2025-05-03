import csv
import os
import json
import random
from faker import Faker
from pathlib import Path

# Initialize Faker for realistic data
fake = Faker()

def generate_schema_metadata(table_name, columns):
    """Generate schema metadata for a table."""
    schema = {
        "table_name": table_name,
        "columns": [
            {
                "name": col["name"],
                "type": col["type"],
                "is_primary_key": col.get("is_primary_key", False),
                "is_foreign_key": col.get("is_foreign_key", False),
                "referenced_table": col.get("referenced_table", ""),
                "referenced_column": col.get("referenced_column", "")
            } for col in columns
        ]
    }
    return schema

def generate_data_for_type(data_type, row_count, referenced_values=None):
    """Generate data for a specific data type."""
    if data_type == "INT":
        if referenced_values:
            return [random.choice(referenced_values) for _ in range(row_count)]
        return [random.randint(1, 10000) for _ in range(row_count)]
    elif data_type == "FLOAT":
        return [round(random.uniform(0.0, 1000.0), 2) for _ in range(row_count)]
    elif data_type == "STRING":
        return [fake.name() if random.choice([True, False]) else fake.address().replace('\n', ', ') for _ in range(row_count)]
    elif data_type == "BOOLEAN":
        return [random.choice([True, False]) for _ in range(row_count)]
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def generate_csv_file(table_name, columns, row_count, output_dir, referenced_data=None):
    """Generate a CSV file and its schema metadata."""
    file_path = os.path.join(output_dir, f"{table_name}.csv")
    schema_metadata = generate_schema_metadata(table_name, columns)
    
    # Generate data
    data = []
    for col in columns:
        ref_values = None
        if col.get("is_foreign_key", False) and referenced_data:
            ref_table = col.get("referenced_table")
            ref_col = col.get("referenced_column")
            ref_values = referenced_data.get(ref_table, {}).get(ref_col, [])
        col_data = generate_data_for_type(col["type"], row_count, ref_values)
        data.append(col_data)
    
    # Write CSV
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        # Write header
        writer.writerow([col["name"] for col in columns])
        # Write data rows
        for row_idx in range(row_count):
            row = [data[col_idx][row_idx] for col_idx in range(len(columns))]
            writer.writerow(row)
    
    return schema_metadata, data

def main():
    # Configuration
    output_dir = "test_csv_files"
    row_count = 100  # Number of rows per CSV
    Path(output_dir).mkdir(exist_ok=True)
    
    # Define test schemas
    schemas = [
        {
            "table_name": "students",
            "columns": [
                {"name": "student_id", "type": "INT", "is_primary_key": True},
                {"name": "name", "type": "STRING"},
                {"name": "gpa", "type": "FLOAT"},
                {"name": "is_active", "type": "BOOLEAN"}
            ]
        },
        {
            "table_name": "products",
            "columns": [
                {"name": "product_id", "type": "INT", "is_primary_key": True},
                {"name": "product_name", "type": "STRING"},
                {"name": "price", "type": "FLOAT"}
            ]
        },
        {
            "table_name": "orders",
            "columns": [
                {"name": "order_id", "type": "INT", "is_primary_key": True},
                {"name": "student_id", "type": "INT", "is_foreign_key": True, "referenced_table": "students", "referenced_column": "student_id"},
                {"name": "product_id", "type": "INT", "is_foreign_key": True, "referenced_table": "products", "referenced_column": "product_id"},
                {"name": "quantity", "type": "INT"},
                {"name": "order_date", "type": "STRING"}
            ]
        }
    ]
    
    # Generate CSV files and collect metadata
    all_metadata = []
    referenced_data = {}
    
    for schema in schemas:
        table_name = schema["table_name"]
        columns = schema["columns"]
        metadata, data = generate_csv_file(table_name, columns, row_count, output_dir, referenced_data)
        
        # Store primary key data for foreign key references
        referenced_data[table_name] = {}
        for col in columns:
            if col.get("is_primary_key", False):
                col_idx = next(i for i, c in enumerate(columns) if c["name"] == col["name"])
                referenced_data[table_name][col["name"]] = data[col_idx]
        
        all_metadata.append(metadata)
    
    # Write schema metadata to JSON
    metadata_file = os.path.join(output_dir, "schemas.json")
    with open(metadata_file, 'w') as f:
        json.dump(all_metadata, f, indent=4)
    
    print(f"Generated CSV files and schema metadata in {output_dir}/")
    print(f"CSV files: {[s['table_name'] + '.csv' for s in schemas]}")
    print(f"Schema metadata: {metadata_file}")

if __name__ == "__main__":
    main()