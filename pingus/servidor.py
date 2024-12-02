from flask import Flask, request, jsonify
import pickle
import numpy as np


classes = ["Adelie", "Chinstrap", "Gentoo"]

def predict_single(penguin, dv, sc, model):
    #Procesar datos categóricos y numéricos
    penguin_categorical = dv.transform([{"island": penguin["island"], "sex": penguin["sex"]}])
    penguin_numerical = sc.transform([[penguin["bill_length_mm"], penguin["bill_depth_mm"],
                                       penguin["flipper_length_mm"], penguin["body_mass_g"]]])
    
    
    penguin_features = np.hstack([penguin_categorical, penguin_numerical])
    print("predict_single para la prediccion:",penguin_features)
    
    #predicción
    y_pred = model.predict(penguin_features)[0]
    y_prob = model.predict_proba(penguin_features)[0][model.classes_.tolist().index(y_pred)]
    return y_pred, y_prob


def predict(dv, sc, model):
    
    penguin = request.get_json()
    species, probability = predict_single(penguin, dv, sc, model)
    
    result = {
        "species": classes[species],
        "probability": float(probability)
    }
    return jsonify(result)


app = Flask("penguin-species")


@app.route("/predict_lr", methods=["POST"])
def predict_lr():
    with open("modelos\\lr.pck", "rb") as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)


@app.route("/predict_svm", methods=["POST"])
def predict_svm():
    with open("modelos\\svm.pck", "rb") as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)


@app.route("/predict_dt", methods=["POST"])
def predict_dt():
    with open("modelos\\dt.pck", "rb") as f:
        dv, model = pickle.load(f)  # Árbol de decisión no usa el escalador
    return predict(dv, None, model)


@app.route("/predict_knn", methods=["POST"])
def predict_knn():
    with open("modelos\\knn.pck", "rb") as f:
        dv, sc, model = pickle.load(f)
    return predict(dv, sc, model)


if __name__ == "__main__":
    app.run(debug=True, port=8000)