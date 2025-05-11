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
        'requests': 'requests'
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
        HumanMessage(content=f"Code: {code}\nImport check result: {import_check_result}"),
    ]

    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"python_repl": python_repl}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"]["code"])
        # Append import check results to the error message
        if import_check_result and "Potential missing imports" in import_check_result:
            state["error_message"] = f"{tool_output}\n\nPotential missing imports detected:\n{import_check_result}"
        else:
            state["error_message"] = tool_output
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
    
    # Check if there are missing imports in the error message
    has_missing_imports = "Potential missing imports" in error
    
    messages = [
        SystemMessage(
            content="""You are a Python code fixer. Analyze the following code and error provided in the usermessage. 
            Your task is to fix that code and provide the correct new code. 
            
            Pay special attention to:
            1. Missing imports - Always add necessary imports at the top of the file
            2. Undefined variables
            3. Syntax errors
            4. Logic errors
            
            VERY IMPORTANT: ONLY RETURN THE UPDATED CODE, NOTHING ELSE! Dont use a markdown style, just the code as Text"""
        ),
        HumanMessage(content=f"Code: {code} | Error: {error}"),
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