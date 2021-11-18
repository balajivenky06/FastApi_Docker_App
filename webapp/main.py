from fastapi import FastAPI, Response
from pydantic import BaseModel

import numpy as np
import torch
from transformers import RobertaTokenizer
import onnxruntime

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
session = onnxruntime.InferenceSession("roberta-sequence-classification-9.onnx")

app = FastAPI()

class Body(BaseModel):
    phrase: str

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy() 
    return tensor.cpu().numpy()

@app.get('/')
def root():
    """curl localhost:8000"""
    return {"message": "This is the root level of our application"}
    
@app.post('/predict')
def predict(body: Body):
    """curl -X POST --data '{"phrase": "Fastapi is awesome"}' localhost:8000/predict"""
    
    input_ids = torch.tensor(tokenizer.encode(body.phrase, add_special_tokens=True)).unsqueeze(0)
    inputs = {session.get_inputs()[0].name: to_numpy(input_ids)}
    out = session.run(None, inputs)
    result = np.argmax(out)
    return {"positive": bool(result)}



