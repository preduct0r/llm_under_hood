import openai
import sqlite3
import os
import json
import argparse
from copy import deepcopy
import datetime  # Added for timestamp

import sys # Added
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_schema import get_schema_from_sqlite_schema
from langfuse import Langfuse
from langfuse.decorators import observe
# from langfuse.openai import openai # OpenAI integration
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


# @observe()
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

# @observe()
def generate_python_code(question: str, db_schema: str) -> str:
    """Generate Python code using LLM based on baseline prompt"""
    
    # Load the baseline prompt template
    with open("practice/text-to-code/prompt.txt", "r") as file:
        baseline_prompt = file.read()
    
    prompt = f"""{baseline_prompt}

Database schema:
{db_schema}

User request: {question}

Respond with only the Python code, without any additional explanations.
"""
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    
    python_code = response.choices[0].message.content.strip()
    
    # Remove markdown code block if present
    if python_code.startswith("```") and python_code.endswith("```"):
        # Remove the first line and last line
        code_lines = python_code.split("\n")
        if code_lines[0].startswith("```python"):
            python_code = "\n".join(code_lines[1:-1])
        else:
            python_code = "\n".join(code_lines[1:-1])
    elif python_code.startswith("```"):
        # Just remove the first line if no closing backticks
        code_lines = python_code.split("\n")
        if code_lines[0].startswith("```python"):
            python_code = "\n".join(code_lines[1:])
        else:
            python_code = "\n".join(code_lines[1:])
    
    return python_code

# @observe()
def fix_code_with_langgraph(file_path: str, max_iterations: int = 5) -> bool:
    """Fix the generated code using the code_fixer LangGraph workflow"""
    from langgraph.graph import StateGraph
    from code_fixer import build_graph

    # Build the graph with tracing enabled and pass max_iterations
    graph = build_graph(max_iterations=max_iterations)
    
    # Run the workflow with the file path
    try:
        result = graph.invoke({
            "message": f"Please analyze the {file_path} file",
            "iterations": 1,  # Starting iterations
            "error": "False",  # Initialize as string to match code_fixer's expectations
            "error_message": "",
            "file_path": file_path,  # Explicitly set the file_path in the initial state
            "code": ""  # Add the missing 'code' field required by AgentState
        })
        
        # Check if the code was fixed successfully
        error_status = result.get("error")
        if error_status is None:
            print(f"Warning: 'error' field missing from result: {result}")
            return False
        return error_status == "False"
    except Exception as e:
        print(f"Error during code fixing: {str(e)}")
        return False

def main(db_path_num: int, question: str, output_folder: str = None, fix_errors: bool = False):
    # Extract the database schema
    db_path = f"practice/pr1/database/org_structure_db{db_path_num}.sqlite"
    schema_info = get_schema_for_db(db_path)

    print(f"Question: {question}")
    
    # Identify relevant table for this question
    relevant_table = identify_relevant_tables(question, schema_info)
    print(f"Relevant table: {relevant_table}")
    
    # Get schema for only the relevant table
    relevant_schema = get_relevant_schema(relevant_table, schema_info)
    
    # Generate Python code
    generated_code = generate_python_code(question, relevant_schema)
    print(f"Generated Python code:\n{generated_code}")
    
    # Save the results if output folder is specified
    if output_folder:
        # Create the output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Create a filename based on the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_folder, f"generated_code_{timestamp}.py") 
        
        with open(output_file, "w") as f:
            f.write(generated_code)
        print(f"Code saved to {output_file}")
        
        # Fix code if requested
        if fix_errors:
            print("Attempting to fix any errors in the generated code...")
            success = fix_code_with_langgraph(output_file)
            if success:
                print(f"Code successfully fixed and saved to {output_file}")
            else:
                print(f"Unable to fully fix code after multiple attempts. Check {output_file} for the latest version.")
    
    return generated_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Python code using LLM')
    parser.add_argument('--db-path-num', type=int, required=True,
                        help='Database number to use')
    parser.add_argument('--question', type=str, required=True,
                        help='Natural language question to generate code for')
    parser.add_argument('--output-folder', type=str, 
                        help='Path to save the generated code (optional)')
    parser.add_argument('--fix-errors', action='store_true',
                        help='Use code_fixer to iteratively fix errors in the generated code')
    args = parser.parse_args()
    
    main(args.db_path_num, args.question, args.output_folder, args.fix_errors)

