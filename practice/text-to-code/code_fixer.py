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
1
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
    model = ChatOpenAI()
    messages = [
        SystemMessage(
            content="""Your task is to evaluate the userinput and extract the filename he provided.
                      ONLY return the last filename, nothing else!"""
        ),
        HumanMessage(content=message),
    ]
    result = model.invoke(messages)
    state["file_path"] = result.content
    return state

def execute_code_with_model(state: AgentState):
    try:
        code = read_code_from_file(state["file_path"])
        state["code"] = code  # Store the code in the state
    except FileNotFoundError:
        state["error"] = "True"
        state["error_message"] = f"File {state['file_path']} not found."
        return state

    model = ChatOpenAI()
    model_with_tools = model.bind_tools([python_repl])

    messages = [
        SystemMessage(
            content=""" You have got the task to execute code. Use the python_repl tool to execute it. I will a message and your task is to detect if it was successfully run or produced an error.
            If the code produced an error just return 'True'. If it was sucessfully executed, return 'False'"""
        ),
        HumanMessage(content=code),
    ]

    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"python_repl": python_repl}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"]["code"])
        state["error_message"] = tool_output
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    result = model_with_tools.invoke(messages)
    print("EVALUATION RESULT:", result)
    state["error"] = result.content
    return state

def rewrite_code(state: AgentState):
    code = state["code"]
    error = state["error_message"]
    state["iterations"] += 1
    model = ChatOpenAI()
    messages = [
        SystemMessage(
            content="You can to analyze the following code and error provided in the usermessage. Your task is to fix that code and provide the user the correct new code. VERY IMPORTANT: ONLY RETURN THE UPDATED CODE, NOTHING ELSE! Dont use a markdown style, just the code as Text"
        ),
        HumanMessage(content=f"Code: {code} | Error: {error}"),
    ]
    ai_msg = model.invoke(messages)
    print("NEW SUGGESTED CODE:", ai_msg.content)
    write_code_to_file(file_path=f'{state["file_path"]}', code=ai_msg.content)
    state["code"] = ai_msg.content
    return state

def next_step(state: AgentState):
    if state["iterations"] > 3:
        print("Max Iterations done.... Exit Agent")
        return "max_iterations"
    if state["error"] == "True":
        print(f"Error in {state['file_path']}. {state['iterations']} tries done")
        return "error"
    if state["error"] == "False":
        print(
            f"Code was probably fixed... check out {state['file_path']} if it is correct"
        )
        return "ok"

# Build and compile the graph with tracing
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("identify_filepath", identify_filepath)
    workflow.add_node("execute_code_with_model", execute_code_with_model)
    workflow.add_node("rewrite_code", rewrite_code)
    
    # Set entry point and add edges
    workflow.set_entry_point("identify_filepath")
    workflow.add_edge("identify_filepath", "execute_code_with_model")
    
    workflow.add_conditional_edges(
        "execute_code_with_model",
        next_step,
        {"error": "rewrite_code", "ok": END, "max_iterations": END},
    )
    workflow.add_edge("rewrite_code", "execute_code_with_model")
    
    # Compile with LangSmith tracing enabled (name parameter is supported)
    return workflow.compile(name="CodeFixerAgent")

if __name__ == "__main__":
    # Build the graph with tracing enabled
    graph = build_graph()
    
    # Run the app with specified project name in the config
    from langchain.callbacks.tracers import LangChainTracer
    from langchain_core.tracers import ConsoleCallbackHandler
    
    # Create LangSmith tracer callback
    tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT", "langraph_code_fixer"))
    
    # Run with tracing enabled
    result = graph.invoke(
        {"message": "Please analyze the /home/den/projects/LangGraph-Tutorial/complex_testscript.py file", "iterations": 1},
        config={"callbacks": [tracer, ConsoleCallbackHandler()]}
    )
    
    # Wait for all traces to be sent
    wait_for_all_tracers()
    
    print("\nFinal result:", result)
    print(f"\nLangSmith trace URL: https://smith.langchain.com/projects/{os.getenv('LANGCHAIN_PROJECT', 'langraph_code_fixer')}")

# Expose the graph at module level for langgraph dev
graph = build_graph() 