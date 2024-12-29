import io
import pickle
import random
import uvicorn
import numpy as np
import pandas as pd
from typing import List, Dict
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File


app = FastAPI()

with open('hw1_best_model.pkl', 'rb') as f:
    model = pickle.load(f)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def normalize_features(data):
    data = data.rename({"mileage": "fuel_consumption", "engine": "engine_vol"}, axis=1)
    # Будем считать, что литр топлива весит 1 кг
    data["fuel_consumption"] = data["fuel_consumption"].str.replace(" kmpl", "").str.replace(" km/kg", "").astype(float)
    data["max_power"] = data["max_power"].str.replace(" bhp", "").replace("", np.nan).astype(float)
    data["engine_vol"] = data["engine_vol"].str.replace(" CC", "").astype(float)

    # Обороты не берем, умножаем на 10 где kgm (надо же как-то в ньютоны перевести)
    # Показал один из способов как обработать
    data["torque_measure"] = data["torque"].str.lower().str.extract(r"(kgm|nm)")
    data["torque"] = data["torque"].str.extract(r"([0-9.]+)").astype(float)
    data.loc[data["torque_measure"] == "kgm", "torque"] *= 10
    data = data.drop(["torque_measure", "torque"], axis=1) # просят удалить
    return data


def get_pandas_data(items: List[Dict]):
    keys = items[0].keys()
    features = {k: [features[k] for features in items] for k in keys}
    data = pd.DataFrame(features)
    return data


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = get_pandas_data([item.model_dump()])
    pred = model.predict(normalize_features(data))
    return pred[0]


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    data = get_pandas_data([item.model_dump() for item in items.objects])
    pred = model.predict(normalize_features(data))
    return list(pred)


@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    contents = file.file.read()
    buffer = io.BytesIO(contents)

    data = pd.read_csv(buffer)
    predict = model.predict(normalize_features(data))
    data["selling_price_predict"] = predict
    data.to_csv("prediction_data.csv", index=False)

    buffer.close()
    file.file.close()

    return FileResponse(
        path='prediction_data.csv',
        filename='prediction_data.csv',
        media_type='multipart/form-data',
    )


if __name__ == "__main__":
    uvicorn.run("hw1_fastapi:app", host="localhost", port=8000, reload=True)
