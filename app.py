from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


from dotenv import load_dotenv

from langchain.agents import initialize_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, MessagesState
from langchain.agents.agent_types import AgentType

import uuid
from utils.tools import find_teacher_cabin, get_fy_event_by_date, get_sy_event_by_date

from langchain_groq import ChatGroq  

model = ChatGroq(temperature=0, model_name="llama3-8b-8192")

load_dotenv()


llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)
# Initialize model + tools + agent

tools = [find_teacher_cabin, get_fy_event_by_date, get_sy_event_by_date]

agent = initialize_agent(
    tools,
    model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)
from langchain.agents import initialize_agent

agent = initialize_agent(
    tools=tools,
    agent_type="zero-shot-react-description",
    llm=llm,
    verbose=True,
    max_iterations=10,  # Increase the max iteration count
    max_execution_time=60,  # Increase the timeout in seconds
)

from langchain_core.messages import AIMessage

def call_model(state: MessagesState):
    result = agent.invoke(state["messages"])
    
    # Extract actual message text from dict
    content = result["output"] if isinstance(result, dict) and "output" in result else str(result)

    print("✅ Parsed agent output:", content)
    return {
        "messages": state["messages"] + [AIMessage(content=content)]
    }

# LangGraph setup
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
memory = MemorySaver()
LLMapp = workflow.compile(checkpointer=memory)

# FastAPI app setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SHARED_THREAD_ID = "shared-context"
ADMIN_MESSAGES = []

@app.get("/", response_class=HTMLResponse)
async def read_user(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/admin/query")
async def admin_query_model(request: Request):
    global ADMIN_MESSAGES
    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            raise ValueError("Missing query")

        config = {"configurable": {"thread_id": SHARED_THREAD_ID}}
        input_messages = [HumanMessage(content=query)]
        output = LLMapp.invoke({"messages": input_messages}, config)

        ADMIN_MESSAGES = output["messages"]
        response_message = output["messages"][-1].content
        return {"response": response_message}

    except Exception as e:
        print(f"[ADMIN ERROR] {e}")
        return {"response": f"Internal Server Error: {str(e)}"}

@app.post("/query")
async def query_model(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        session_id = data.get("session_id") or str(uuid.uuid4())
        user_thread_id = f"user-{session_id}"

        input_messages = ADMIN_MESSAGES + [HumanMessage(content=query)] if ADMIN_MESSAGES else [HumanMessage(content=query)]
        config = {"configurable": {"thread_id": user_thread_id}}

        output = LLMapp.invoke({"messages": input_messages}, config)
        response_message = output["messages"][-1].content
        return {"response": response_message}

    except Exception as e:
        print(f"[USER ERROR] {e}")
        return {"response": f"Internal Server Error: {str(e)}"}

@app.post("/admin/reset")
def reset_admin_context():
    global ADMIN_MESSAGES
    ADMIN_MESSAGES = []
    return {"message": "Admin context cleared."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)
