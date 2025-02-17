import dash
from dash import dcc, html, Input, Output
import pickle
import numpy as np
import plotly.express as px

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Dash app
app = dash.Dash(__name__)

server=app.server

app.layout = html.Div([
    html.H3("Simple AI Model Prediction"),
    
    html.Label("Feature 1:"),
    dcc.Input(id="input-1", type="number", placeholder="Enter value", step=0.1),
    
    html.Label("Feature 2:"),
    dcc.Input(id="input-2", type="number", placeholder="Enter value", step=0.1),
    
    html.Button("Predict", id="predict-btn", n_clicks=0),
    
    html.Div(id="prediction-output", style={'fontSize': 20, 'marginTop': 20}),
    
    dcc.Graph(id="confidence-graph")
])

@app.callback(
    [Output("prediction-output", "children"),
     Output("confidence-graph", "figure")],
    [Input("predict-btn", "n_clicks")],
    [dash.State("input-1", "value"),
     dash.State("input-2", "value")]
)
def make_prediction(n_clicks, feature1, feature2):
    if feature1 is None or feature2 is None:
        return "Enter valid values!", px.bar()

    # Convert input to numpy array
    input_data = np.array([[feature1, feature2]])
    
    # Get model prediction
    predicted_class = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]  # Confidence scores

    # Prepare bar chart
    fig = px.bar(x=["Class 0", "Class 1"], y=probabilities, labels={'x': 'Class', 'y': 'Probability'})

    return f"Predicted Class: {predicted_class}", fig

if __name__ == "__main__":
    app.run_server(debug=True)
