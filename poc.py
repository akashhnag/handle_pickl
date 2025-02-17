import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from sklearn.linear_model import LinearRegression

# Initialize Dash app
app = dash.Dash(__name__)

# Sample AI model (Linear Regression)
x_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_train = np.array([2, 4, 6, 8, 10])
model = LinearRegression()
model.fit(x_train, y_train)

# Layout
app.layout = html.Div([
    html.H1("AI Model Visualization with Dash"),
    dcc.Slider(
        id='input-slider',
        min=0, max=10, step=0.1,
        value=5,
        marks={i: str(i) for i in range(11)}
    ),
    html.Div(id='prediction-output', style={'fontSize': 24, 'marginTop': 20}),
    dcc.Graph(id='prediction-graph')
])

# Callback for updating prediction and graph
@app.callback(
    [Output('prediction-output', 'children'),
     Output('prediction-graph', 'figure')],
    [Input('input-slider', 'value')]
)
def update_output(value):
    prediction = model.predict(np.array([[value]]))[0]
    
    # Generate graph
    x_values = np.linspace(0, 10, 100).reshape(-1, 1)
    y_values = model.predict(x_values)
    
    figure = {
        'data': [
            go.Scatter(x=x_values.flatten(), y=y_values, mode='lines', name='Model Prediction'),
            go.Scatter(x=[value], y=[prediction], mode='markers', marker={'color': 'red', 'size': 10}, name='Input Point')
        ],
        'layout': go.Layout(title='AI Prediction Visualization', xaxis={'title': 'Input'}, yaxis={'title': 'Output'})
    }

    return f'Predicted Value: {prediction:.2f}', figure

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
