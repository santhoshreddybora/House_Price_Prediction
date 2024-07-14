
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from HPP.constants import APP_HOST, APP_PORT
from HPP.pipeline.prediction_pipeline import HPPData, HppClassifier
from HPP.pipeline.training_pipeline import TrainingPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.location: Optional[str] = None
        self.no_of_BHK: Optional[str] = None
        self.total_sqft: Optional[str] = None
        self.bath: Optional[str] = None
        

    async def get_usvisa_data(self):
        form = await self.request.form()
        self.location = form.get("location")
        self.no_of_BHK = form.get("no_of_BHK")
        self.total_sqft = form.get("total_sqft")
        self.bath = form.get("bath")


@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "hpp.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainingPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_usvisa_data()
        
        Hpp_data = HPPData(
                                location = form.location,
                                no_of_BHK = form.no_of_BHK,
                                total_sqft= form.total_sqft,
                                bath= form.bath
                               )
        
        hpp_df = Hpp_data.get_hpp_input_data_frame()

        model_predictor = HppClassifier()


        value = model_predictor.predict(dataframe=hpp_df)[0]
        value=round(value,2)
        status = None
        if value != 0:
            status = f"House price for above features is {value}Lakhs INR"
        else:
            status = f"House price is not able to retrieve "

        return templates.TemplateResponse(
            "hpp.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)