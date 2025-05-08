import sqlite3
import os # For creating/deleting a demo file
import argparse


def get_schema_from_sqlite_schema(db_path):
    """Gets all CREATE statements from the sqlite_schema table."""
    schema_info = {}
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Fetch type, name, and the original SQL command
            cursor.execute("""
                SELECT type, name, sql
                FROM sqlite_schema
                WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%'
                ORDER BY type, name;
            """)
            results = cursor.fetchall()
            for type, name, sql in results:
                if type not in schema_info:
                    schema_info[type] = {}
                schema_info[type][name] = sql
    except sqlite3.Error as e:
        print(f"Database error (sqlite_schema): {e}")
    return schema_info


def main(db_file):
    """Main function to demonstrate schema extraction."""
    print("## Method 1: Using sqlite_schema ##")
    full_schema = get_schema_from_sqlite_schema(db_file)
    if full_schema:
        for obj_type, items in full_schema.items():
            print(f"\n### {obj_type.capitalize()}s:")
            for name, sql in items.items():
                print(f"-- Schema for {obj_type} '{name}' --")
                print(sql)
    else:
        print("Could not retrieve schema information.")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract schema information from SQLite database.')
    parser.add_argument('--db_path', '-d', 
                        help='Path to the SQLite database file')
    args = parser.parse_args()
    
    main(args.db_path)


