import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("models/model.onnx")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def predict_survival(features: list):

    input_array = np.array([features], dtype=np.float32).reshape(1, -1)
    outputs = session.run([output_name], {input_name: input_array})[0]

    prediction = outputs[0] 

    if prediction[0] < 0.5:
        return 0  
    else:
        return 1

if __name__ == "__main__":
    features = [3, 0, 22, 1, 0, 7.25] 
    result = predict_survival(features)
    print(result)

