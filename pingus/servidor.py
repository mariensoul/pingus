from flask import Flask, request, jsonify
import pickle

# Lista de clases (especies de pingüinos)
classes = ['Adelie', 'Chinstrap', 'Gentoo']

# Predicción para una sola entrada
def predict_single(pingu, sc, model):
    features = [[
        pingu["bill_length_mm"],
        pingu["bill_depth_mm"],
        pingu["flipper_length_mm"],
        pingu["body_mass_g"],
        pingu["island_Biscoe"],
        pingu["island_Dream"],
        pingu["island_Torgersen"],
        pingu["sex_Female"],
        pingu["sex_Male"]
    ]]
    features_std = sc.transform(features)
    
    #Predicción de la clase y probabilidad
    y_pred = model.predict(features_std)[0]
    y_prob = model.predict_proba(features_std)[0][y_pred]
    return y_pred, y_prob


def predict(sc, model):
    
    pingu = request.get_json()
    
    # Llamar a predict_single para obtener predicción y probabilidad
    especie, probabilidad = predict_single(pingu, sc, model)
    
    result = {
        'especie': classes[especie],
        'probabilidad': float(probabilidad)
    }
    return jsonify(result)


app = Flask('pingu')

# Rutas para cada modelo
@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('modelos/lr.pck', 'rb') as f:
        sc, model = pickle.load(f)
    return predict(sc,model)

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('modelos/svm.pck', 'rb') as f:
        sc, model = pickle.load(f)
    return predict(sc,model)

@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    with open('modelos/dt.pck', 'rb') as f:
        sc, model = pickle.load(f)
    return predict(sc,model)

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('modelos/knn.pck', 'rb') as f:
        sc, model = pickle.load(f)
    return predict(sc,model)


# Ejecutar la app
if __name__ == "__main__":
    app.run(debug=True, port=8000)
