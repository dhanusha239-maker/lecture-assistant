from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from agent import ask_agent

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/ask")
def ask(q: str):
    try:
        answer = ask_agent(q)
        return {"answer": answer}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )