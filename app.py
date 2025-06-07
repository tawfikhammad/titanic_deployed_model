from fastapi import FastAPI 
from pydantic import BaseModel
from typing import List
from inference import predict_survival

app = FastAPI()
class FeaturesInput(BaseModel):
    features: List[float]

@app.post("/predict")
def predict(input_data: FeaturesInput):
    features = input_data.features
    result = predict_survival(features)
    return {
        "prediction": result,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

