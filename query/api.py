from fastapi import FastAPI

from query.agent import agent

app = FastAPI()
agent_instance = agent()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query/{user_query}")
async def query_knowledge_base(user_query: str):
    response = await agent_instance.chat(user_query)
    print(response)
