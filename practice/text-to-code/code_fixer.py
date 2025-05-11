import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langsmith import Client
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
import sys # Added
import re
import sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Run the app with specified project name in the config
from langchain.callbacks.tracers import LangChainTracer
from langchain_core.tracers import ConsoleCallbackHandler

# Load environment variables
load_dotenv()

# Initialize LangSmith client
client = Client()

# Helper functions for file I/O
def read_code_from_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        code = file.read()
    return code

def write_code_to_file(file_path: str, code: str):
    with open(file_path, "w") as file:
        file.write(code)

# Initialize Python REPL tool
repl = PythonREPL()

@tool
def python_repl(code: str):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
        print("RESULT CODE EXECUTION:", result)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Executed:\n```python\n{code}\n```\nStdout: {result}"

# Helper function to extract and validate SQL queries
def extract_sql_queries(code: str):
    """Extract SQL queries from Python code and check if they're valid."""
    # Find all triple-quoted strings which might contain SQL
    sql_matches = re.findall(r'(?:\"\"\"|\'\'\')(.*?)(?:\"\"\"|\'\'\')|\bquery\s*\(\s*[\"\'](.*?)[\"\']', code, re.DOTALL)
    sql_queries = []
    
    # Flatten the matches and remove empty strings
    for match in sql_matches:
        for query in match:
            if query.strip() and ('SELECT' in query.upper() or 'INSERT' in query.upper() or 
                                 'UPDATE' in query.upper() or 'DELETE' in query.upper()):
                sql_queries.append(query.strip())
    
    return sql_queries

# Helper function to validate SQL queries against available databases
def validate_sql_queries(queries):
    """Validate SQL queries against available databases."""
    results = []
    
    # Try to find database files
    db_paths = []
    for i in range(1, 4):  # Try databases 1-3
        db_path = f"practice/pr1/database/org_structure_db{i}.sqlite"
        if os.path.exists(db_path):
            db_paths.append(db_path)
    
    if not db_paths:
        return ["No database files found to validate SQL queries"]
    
    # Validate each query against each database
    for db_path in db_paths:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for query in queries:
                try:
                    # Just validate the query without executing it fully
                    cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                    results.append(f"Query valid in {db_path}: {query[:50]}...")
                except sqlite3.Error as e:
                    results.append(f"Error in {db_path} for query: {query[:50]}...: {str(e)}")
            
            conn.close()
        except sqlite3.Error as e:
            results.append(f"Failed to connect to {db_path}: {str(e)}")
    
    return results

# Define the state type
class AgentState(TypedDict):
    message: str
    error: bool
    error_message: str
    file_path: str
    code: str
    iterations: int

# Node functions
def identify_filepath(state: AgentState):
    message = state["message"]
    
    # Extract the filepath directly from the message if it contains "analyze the"
    if "analyze the " in message and " file" in message:
        filepath = message.split("analyze the ")[1].split(" file")[0]
        state["file_path"] = filepath
        return state
    
    # Fallback to using the LLM if direct extraction fails
    model = ChatOpenAI(model="gpt-4o-mini")
    messages = [
        SystemMessage(
            content="""Your task is to evaluate the userinput and extract the complete filepath he provided.
                      ONLY return the filepath, nothing else!"""
        ),
        HumanMessage(content=message),
    ]
    result = model.invoke(messages)
    state["file_path"] = result.content
    return state

def execute_code_with_model(state: AgentState):
    try:
        # First try to read the file with the path as is
        try:
            code = read_code_from_file(state["file_path"])
        except FileNotFoundError:
            # If that fails, try to extract the full path from the message
            message = state["message"]
            if "analyze the " in message:
                full_path = message.split("analyze the ")[1].split(" file")[0]
                code = read_code_from_file(full_path)
                # Update the file_path to use the full path
                state["file_path"] = full_path
            else:
                raise FileNotFoundError(f"File {state['file_path']} not found.")
        
        state["code"] = code  # Store the code in the state
    except FileNotFoundError as e:
        state["error"] = "True"
        state["error_message"] = str(e)
        return state

    model = ChatOpenAI()
    model_with_tools = model.bind_tools([python_repl])

    # First, let's check for potential missing imports
    # Create a temporary file for the code to analyze
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name
    
    import_check_code = f"""
import ast
import re

def find_potential_missing_imports(file_path):
    with open('{temp_file_path}', 'r') as f:
        code_str = f.read()
    
    # Parse the code into an AST
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return f"Syntax error in code: {{str(e)}}"
    
    # Find all names that might be undefined
    undefined_names = set()
    imported_names = set()
    defined_names = set()
    
    # First pass: collect all imported and defined names
    for node in ast.walk(tree):
        # Check for imports
        if isinstance(node, ast.Import):
            for name in node.names:
                imported_names.add(name.name)
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                imported_names.add(name.name)
        # Check for function definitions
        elif isinstance(node, ast.FunctionDef):
            defined_names.add(node.name)
        # Check for class definitions
        elif isinstance(node, ast.ClassDef):
            defined_names.add(node.name)
        # Check for variable assignments
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
    
    # Second pass: find all used names
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in imported_names and node.id not in defined_names and not node.id in __builtins__:
                undefined_names.add(node.id)
    
    # Common libraries that might be missing
    common_libraries = {{
        'np': 'numpy',
        'pd': 'pandas',
        'plt': 'matplotlib.pyplot',
        'sns': 'seaborn',
        'tf': 'tensorflow',
        'torch': 'torch',
        'sk': 'sklearn',
        'os': 'os',
        'sys': 'sys',
        'json': 'json',
        're': 're',
        'math': 'math',
        'datetime': 'datetime',
        'requests': 'requests',
        'sqlite3': 'sqlite3'
    }}
    
    # Check for potential missing imports
    missing_imports = []
    for name in undefined_names:
        if name in common_libraries:
            missing_imports.append(f"import {{common_libraries[name]}} as {{name}}")
        elif name in ['pyplot', 'plt']:
            missing_imports.append("import matplotlib.pyplot as plt")
        elif name == 'numpy':
            missing_imports.append("import numpy as np")
        elif name == 'pandas':
            missing_imports.append("import pandas as pd")
        elif name == 'query':
            missing_imports.append("# 'query' function is undefined - might need to implement a database query function")
    
    return missing_imports

# Analyze the code
missing_imports = find_potential_missing_imports('{temp_file_path}')

if missing_imports:
    print("Potential missing imports:")
    for imp in missing_imports:
        print(f"- {{imp}}")
"""

    # Execute the import check
    import_check_result = repl.run(import_check_code)
    
    # Check for SQL queries in the code
    sql_queries = extract_sql_queries(code)
    sql_validation_results = []
    
    # Check for SQL schema issues
    sql_schema_check = ""
    if sql_queries:
        print(f"Found {len(sql_queries)} SQL queries in the code")
        sql_validation_results = validate_sql_queries(sql_queries)
        
        # Additional schema validation for common tables
        schema_check_code = f"""
import sqlite3
import os

def check_sql_schema_compatibility(queries):
    results = []
    
    # Path to the database
    db_path = "practice/text-to-sql/database/org_structure_db1.sqlite"
    
    if not os.path.exists(db_path):
        alt_paths = [
            "../text-to-sql/database/org_structure_db1.sqlite",
            "../../text-to-sql/database/org_structure_db1.sqlite",
            "/home/den/projects/llm_under_hood/practice/text-to-sql/database/org_structure_db1.sqlite"
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                db_path = alt_path
                break
        else:
            return ["Could not find database file"]
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        results.append(f"Available tables: {{', '.join(tables)}}")
        
        # Common misnamed tables
        common_misnames = {{"Department": "Departments", "Employee": "Employees"}}
        
        for query in queries:
            query_upper = query.upper()
            for misname, correct in common_misnames.items():
                if f" {{misname.upper()}} " in f" {{query_upper}} " and correct.upper() in [t.upper() for t in tables]:
                    results.append(f"Warning: '{{misname}}' should be '{{correct}}' in query")
            
            # Check for column existence in tables
            for table in tables:
                if f" {{table.upper()}} " in f" {{query_upper}} " or f"FROM {{table.upper()}}" in query_upper:
                    cursor.execute(f"PRAGMA table_info({{table}});")
                    columns = [row[1] for row in cursor.fetchall()]
                    results.append(f"Table {{table}} columns: {{', '.join(columns)}}")
        
        conn.close()
    except sqlite3.Error as e:
        results.append(f"Database error: {{str(e)}}")
    
    return results

# Check SQL queries against schema
sql_queries = {sql_queries}
schema_results = check_sql_schema_compatibility(sql_queries)
for result in schema_results:
    print(result)
"""
        
        try:
            sql_schema_check = repl.run(schema_check_code)
        except Exception as e:
            sql_schema_check = f"Error checking SQL schema: {str(e)}"
    
    # Clean up the temporary file
    try:
        os.remove(temp_file_path)
    except:
        pass

    # Now execute the actual code
    messages = [
        SystemMessage(
            content="""You have got the task to execute code. Use the python_repl tool to execute it. I will send a message and your task is to detect if it was successfully run or produced an error.
            If the code produced an error or has missing imports just return 'True'. If it was successfully executed, return 'False'.
            """
        ),
        HumanMessage(content=f"Code: {code}\nImport check result: {import_check_result}\nSQL validation results: {sql_validation_results}\nSQL schema check: {sql_schema_check}"),
    ]

    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"python_repl": python_repl}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"]["code"])
        # Append import check and SQL validation results to the error message
        error_message_parts = [tool_output]
        
        if import_check_result and "Potential missing imports" in import_check_result:
            error_message_parts.append(f"Potential missing imports detected:\n{import_check_result}")
        
        if sql_validation_results:
            error_message_parts.append(f"SQL validation results:\n{', '.join(sql_validation_results)}")
        
        if sql_schema_check:
            error_message_parts.append(f"SQL schema check:\n{sql_schema_check}")
            
        # Check if 'query' function is used but not defined
        if "name 'query' is not defined" in tool_output:
            error_message_parts.append("The 'query' function is used but not defined. You need to implement a database connection and query function.")
            
        state["error_message"] = "\n\n".join(error_message_parts)
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    result = model_with_tools.invoke(messages)
    print("EVALUATION RESULT:", result)
    state["error"] = result.content
    return state

def rewrite_code(state: AgentState):
    code = state["code"]
    error = state["error_message"]
    file_path = state["file_path"]
    state["iterations"] += 1
    model = ChatOpenAI()
    
    # Extract SQL queries for context
    sql_queries = extract_sql_queries(code)
    sql_context = ""
    if sql_queries:
        sql_context = "SQL queries found in code:\n" + "\n".join([f"- {q[:100]}..." for q in sql_queries])
    
    messages = [
        SystemMessage(
            content="""You are a Python code fixer. Analyze the following code and error provided in the usermessage. 
            Your task is to fix that code and provide the correct new code. 
            
            Pay special attention to:
            1. Missing imports - Always add necessary imports at the top of the file
            2. Undefined variables
            3. Syntax errors
            4. Logic errors
            5. SQL queries - If the code contains SQL queries, make sure to implement a proper query function 
               that connects to a SQLite database and executes the query
            
            If you see a 'query' function being used but not defined, implement it to:
            - Connect to a SQLite database (use 'practice/text-to-sql/database/org_structure_db1.sqlite' as default path)
            - Execute the SQL query and return results as a pandas DataFrame
            - Check for table name accuracy - use 'Departments' (not 'Department') and 'Employees' (not 'Employee')
            - Check for column name accuracy - verify column names match the database schema
            - Add proper error handling and fallback mechanisms
            
            For SQL queries:
            - Validate table names against the actual database schema
            - Ensure JOIN conditions use the correct column names
            - Make sure queries follow SQLite syntax
            
            VERY IMPORTANT: ONLY RETURN THE UPDATED CODE, NOTHING ELSE! Dont use a markdown style, just the code as Text"""
        ),
        HumanMessage(content=f"Code: {code}\nError: {error}\n{sql_context}"),
    ]
    ai_msg = model.invoke(messages)
    print("NEW SUGGESTED CODE:", ai_msg.content)
    write_code_to_file(file_path=file_path, code=ai_msg.content)
    state["code"] = ai_msg.content
    return state

# Build and compile the graph with tracing
def build_graph(max_iterations: int = 3):
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("identify_filepath", identify_filepath)
    workflow.add_node("execute_code_with_model", execute_code_with_model)
    workflow.add_node("rewrite_code", rewrite_code)
    
    # Set entry point and add edges
    workflow.set_entry_point("identify_filepath")
    workflow.add_edge("identify_filepath", "execute_code_with_model")
    
    # Update next_step function to use max_iterations parameter
    def patched_next_step(state: AgentState):
        if state["iterations"] > max_iterations:
            print(f"Max Iterations ({max_iterations}) done.... Exit Agent")
            return "max_iterations"
        if state["error"] == "True":
            print(f"Error in {state['file_path']}. {state['iterations']} tries done")
            return "error"
        if state["error"] == "False":
            print(
                f"Code was probably fixed... check out {state['file_path']} if it is correct"
            )
            return "ok"
    
    workflow.add_conditional_edges(
        "execute_code_with_model",
        patched_next_step,
        {"error": "rewrite_code", "ok": END, "max_iterations": END},
    )
    workflow.add_edge("rewrite_code", "execute_code_with_model")
    
    # Compile with LangSmith tracing enabled (name parameter is supported)
    return workflow.compile(name="CodeFixerAgent")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fix code using LangGraph')
    parser.add_argument('--file_path', type=str, default="practice/text-to-code/testcase.py",
                        help='Path to the file to fix')
    parser.add_argument('--max_iterations', type=int, default=3,
                        help='Maximum number of iterations to try fixing the code')
    args = parser.parse_args()
    
    # Build the graph with tracing enabled
    graph = build_graph(max_iterations=args.max_iterations)
    
    # Create LangSmith tracer callback
    tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT", "langraph_code_fixer"))
    
    # Run with tracing enabled
    result = graph.invoke(
        {"message": f"Please analyze the {args.file_path} file", "iterations": 1},
        config={"callbacks": [tracer, ConsoleCallbackHandler()]}
    )
    
    # Wait for all traces to be sent
    wait_for_all_tracers()
    
    print("\nFinal result:", result)
    print(f"\nLangSmith trace URL: https://smith.langchain.com/projects/{os.getenv('LANGCHAIN_PROJECT', 'langraph_code_fixer')}")

# Expose the graph at module level for langgraph dev
graph = build_graph(max_iterations=3) 