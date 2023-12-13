from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from src.components.predict import Predictor
from src.components.Youtube_Scrapper import Get_Youtube_comments

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("template.html", {"request": request})

# @app.post("/submit")
# def submit(request: Request, input_text: str = Form(...)):
#     return {"request": str(request), "input_text": input_text}

@app.post("/submit")
def submit(request: Request, input_text: str = Form(...)):

    youtube=Get_Youtube_comments()
    df = youtube.IngestData(input_text)
    pred=Predictor()
    df = pred.predict(df)
    return HTMLResponse(content=df.to_html())