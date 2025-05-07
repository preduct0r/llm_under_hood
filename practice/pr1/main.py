import openai
import sqlite3
import os
import json
import argparse
from copy import deepcopy
from get_schema import get_schema_from_sqlite_schema
from langfuse import Langfuse
from langfuse.decorators import observe
from langfuse.openai import openai # OpenAI integration
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


langfuse = Langfuse(
  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host=os.getenv("LANGFUSE_HOST")
)
openai.api_key = os.getenv("OPENAI_API_KEY")  


def get_schema_for_db(db_path: str) -> dict:
    """Extract schema information from the database"""
    return get_schema_from_sqlite_schema(db_path)

def format_schema_string(schema_info: dict) -> str:
    """Format schema info into a string for the prompt"""
    schema_str = ""
    
    if "table" in schema_info:
        for table_name, sql in schema_info["table"].items():
            schema_str += f"{sql};\n\n"
    
    return schema_str


@observe()
def identify_relevant_tables(question: str, schema_info: dict) -> str:
    """Identify the most relevant table to the question"""
    
    # Create schema overview for LLM
    tables_overview = ""
    if "table" in schema_info:
        for table_name, sql in schema_info["table"].items():
            tables_overview += f"{sql};\n\n"
    
    prompt = f"""
    Given the following SQL database structure:
    {tables_overview}
    
    And this natural language question:
    "{question}"
    
    Identify the single most relevant table for answering this question.
    Return only the table name without any additional explanation.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    
    # Process the response to get the table name
    table_name = response.choices[0].message.content.strip()
    return table_name

def get_relevant_schema(relevant_table: str, schema_info: dict) -> str:
    """Extract schema only for the relevant table"""
    relevant_schema = ""
    
    if "table" in schema_info:
        for table_name, sql in schema_info["table"].items():
            # Check if this table is our relevant table
            if table_name.lower() == relevant_table.lower():
                relevant_schema += f"{sql};\n\n"
    
    return relevant_schema

@observe()
def generate_sql(question: str, db_schema: str) -> str:
    """Генерация SQL запроса с помощью LLM"""
    prompt = f"""
    Given the following SQL database structure:
    {db_schema}
    
    Convert this natural language question into a SQL query:
    {question}
    
    Return ONLY the SQL query without any additional explanation.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    
    sql = response.choices[0].message.content.strip()
    
    # Remove markdown code block if present
    if sql.startswith("```") and sql.endswith("```"):
        # Remove the first line and last line
        sql_lines = sql.split("\n")
        sql = "\n".join(sql_lines[1:-1])
    elif sql.startswith("```"):
        # Just remove the first line if no closing backticks
        sql_lines = sql.split("\n")
        sql = "\n".join(sql_lines[1:])
    
    return sql

def execute_sql(sql: str, db_path: str) -> list:
    """Выполнение SQL запроса в SQLite базе"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    finally:
        conn.close()
    
    return results

def main(db_path_num: int, qa_path: str):
    # Extract the database schema
    db_path = f"practice/pr1/database/org_structure_db{db_path_num}.sqlite"
    schema_info = get_schema_for_db(db_path)

    qa_data = json.load(open(qa_path))
    
    for item in qa_data:
        # user_question = input("Введите ваш вопрос о данных: ")
        user_question = item["question"]
        
        # Identify relevant table for this question
        relevant_table = identify_relevant_tables(user_question, schema_info)
        print(f"Релевантная таблица: {relevant_table}")
        
        # Get schema for only the relevant table
        relevant_schema = get_relevant_schema(relevant_table, schema_info)
        
        # Генерация SQL
        generated_sql = generate_sql(user_question, relevant_schema)
        print(f"Сгенерированный SQL: {generated_sql}")
        
        # Выполнение запроса
        try:
            # Use the generated SQL directly
            sql_results = execute_sql(generated_sql, db_path)
            print("\nРезультаты:")
            item["ground_truth"] = deepcopy(item["answer"]) # для проверки
            
            # Format the answer appropriately based on result type
            if len(sql_results) == 1 and len(sql_results[0]) == 1:
                # Single value result - return just the value
                item["answer"] = sql_results[0][0]
            elif len(sql_results) >= 1 and len(sql_results[0]) == 1:
                # List of single values - extract into simple list
                item["answer"] = [row[0] for row in sql_results]
            else:
                # More complex result structure - keep as is
                item["answer"] = sql_results
        except sqlite3.Error as e:
            print(f"Ошибка выполнения запроса: {e}")

        # break
    json.dump(qa_data, open(f"practice/pr1/answers/{db_path_num}.json", "w"), indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Выполнение SQL-запросов с использованием LLM')
    parser.add_argument('--db-path-num', type=int, required=True,
                        help='Путь к файлу базы данных SQLite')
    parser.add_argument('--test-qa', type=str, required=True,
                        help='Тестовые данные')
    args = parser.parse_args()
    
    main(args.db_path_num, args.test_qa)

