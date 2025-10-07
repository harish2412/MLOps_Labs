from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from predict import predict_data

app = FastAPI()

class CancerInput(BaseModel):
    features: list[float] = Field(min_length=30, max_length=30)

class CancerResponse(BaseModel):
    response: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=CancerResponse)
async def predict_breast_cancer(x: CancerInput):
    try:
        probs = predict_data([x.features])   
        label = int(probs[0][1] >= 0.5)      
        return CancerResponse(response=label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))