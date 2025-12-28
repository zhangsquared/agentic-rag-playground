from fastapi import FastAPI

from query.agent import agent

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query/{user_query}")
async def query_knowledge_base(user_query: str):
    response = await agent.chat(user_query)
    print(response)
