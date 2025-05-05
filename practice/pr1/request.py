import openai
import sqlite3
import os
import argparse

# Конфигурация OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")  # Используйте свой API-ключ

def generate_sql(question: str) -> str:
    """Генерация SQL запроса с помощью LLM"""
    prompt = f"""
    Given the following SQL table structure:
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        department TEXT,
        salary INTEGER,
        hire_date DATE
    );
    
    Convert this natural language question into a SQL query:
    {question}
    
    Return ONLY the SQL query without any additional explanation.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return response.choices[0].message.content.strip()

def execute_sql(sql: str, db_path: str = '/path/to/your/existing/database.db') -> list:
    """Выполнение SQL запроса в SQLite базе"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    finally:
        conn.close()
    
    return results

def main(db_path: str):
    # user_question = input("Введите ваш вопрос о данных: ")
    user_question = "Which employee is responsible for maintaining the most systems?"
    
    # Генерация SQL
    generated_sql = generate_sql(user_question)
    print(f"Сгенерированный SQL: {generated_sql}")
    
    # Выполнение запроса
    try:
        generated_sql = "SELECT name \nFROM employees \nWHERE department = 'Systems' \nORDER BY salary DESC \nLIMIT 1;"
        data = execute_sql(generated_sql, db_path)
        print("\nРезультаты:")
        for row in data:
            print(row)
    except sqlite3.Error as e:
        print(f"Ошибка выполнения запроса: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Выполнение SQL-запросов с использованием LLM')
    parser.add_argument('--db-path', type=str, default='/path/to/your/existing/database.db',
                        help='Путь к файлу базы данных SQLite')
    args = parser.parse_args()
    
    main(args.db_path)