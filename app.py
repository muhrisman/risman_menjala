import numpy as np
from joblib import load
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Load your trained models (make sure the paths are correct)
model_sr = load('model_sr.pkl')
model_abw = load('model_abw.pkl')

app = dash.Dash(__name__)

# Default input values
default_values = {
    "days_until_harvest": 60,
    "cycle_age_days": 60,
    "total_seed": 100000,
    "area": 1000,
    "total_shrimp": 95000,
    "total_weight": 1500,
    "feed_quantity": 1200,
    "morning_temperature": 28,
    "evening_temperature": 27,
    "morning_do": 7.2,
    "evening_do": 6.8,
    "morning_salinity": 35,
    "evening_salinity": 34,
    "morning_pH": 7.8,
    "evening_pH": 7.7,
    "nitrate": 0.25,
    "nitrite": 0.02,
    "alkalinity": 120,
    "price_per_kg": 12  # Price per kg in USD
}

# Exchange rate from USD to IDR
usd_to_idr = 15000

# App layout
app.layout = html.Div([
    html.H1("Shrimp Farming Forecast Dashboard", style={'textAlign': 'center', 'marginBottom': '40px'}),
    html.Div([
        # Generate input fields
        html.Div([
            html.Label(f"{key}:", style={'display': 'block', 'textAlign': 'center', 'marginBottom': '5px'}),
            dcc.Input(id=key, type='number', value=value, style={'width': '100%', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'})
        ], style={'flex': '1', 'margin': '10px', 'minWidth': '200px'})
        for key, value in default_values.items()
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),
    
    html.Div([
        html.Button('Submit', id='submit-val', n_clicks=0, style={
            'backgroundColor': '#007bff', 
            'color': 'white', 
            'padding': '15px 30px', 
            'fontSize': '16px', 
            'border': 'none', 
            'borderRadius': '5px',
            'cursor': 'pointer',
            'marginTop': '20px'
        }),
    ], style={'textAlign': 'center'}),

    html.Hr(style={'marginTop': '40px', 'marginBottom': '40px'}),
    
    html.Div([
        dcc.Graph(id='survival-rate-plot', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='abw-plot', style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
    
    html.Div([
        dcc.Graph(id='biomass-plot', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='revenue-plot', style={'width': '48%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'flexWrap': 'wrap'}),
    
    html.Div(id='revenue-output', style={'textAlign': 'center', 'marginTop': '40px', 'fontSize': '18px', 'fontWeight': 'bold'})
])

# Callback for updating the output
@app.callback(
    [
        Output('survival-rate-plot', 'figure'),
        Output('abw-plot', 'figure'),
        Output('biomass-plot', 'figure'),
        Output('revenue-plot', 'figure'),
        Output('revenue-output', 'children')
    ],
    [Input('submit-val', 'n_clicks')],
    [Input(key, 'value') for key in default_values]
)
def update_output(n_clicks, *input_values):
    inputs = {key: int(val) if val is not None else default_values[key] for key, val in zip(default_values.keys(), input_values)}

    # Generate days array
    days = np.arange(1, inputs['days_until_harvest'] + 1)

    # Prepare features for prediction across all days for SR and ABW models
    features_sr = np.array([
        [
            day, 
            inputs['total_seed'], 
            inputs['area'], 
            inputs['total_shrimp'], 
            inputs['total_weight'],
            inputs['feed_quantity'], 
            inputs['morning_temperature'], 
            inputs['evening_temperature'],
            inputs['morning_do'], 
            inputs['evening_do'],
            inputs['morning_salinity'], 
            inputs['evening_salinity'],
            inputs['morning_pH'], 
            inputs['evening_pH'],
            inputs['nitrate'], 
            inputs['nitrite'], 
            inputs['alkalinity']
        ] for day in days
    ])
    sr_predictions = model_sr.predict(features_sr)

    features_abw = np.array([
        [day, inputs['area'], inputs['feed_quantity']] for day in days
    ])
    abw_predictions = model_abw.predict(features_abw)
    
    daily_biomass = abw_predictions / 1000 * inputs['total_shrimp']
    cumulative_biomass = np.cumsum(daily_biomass)
    cumulative_revenue_idr = np.cumsum(daily_biomass * inputs['price_per_kg'] * usd_to_idr)

    sr_figure = go.Figure(data=[go.Scatter(x=days, y=sr_predictions, mode='lines', name='Survival Rate', line=dict(color='blue'))], layout=go.Layout(title="Predicted Survival Rate Over Time", xaxis={'title': "Days"}, yaxis={'title': "Survival Rate (%)"}))
    abw_figure = go.Figure(data=[go.Scatter(x=days, y=abw_predictions, mode='lines', name='Average Body Weight', line=dict(color='green'))], layout=go.Layout(title="Predicted Average Body Weight Over Time", xaxis={'title': "Days"}, yaxis={'title': "Weight (grams)"}))
    biomass_figure = go.Figure(data=[go.Scatter(x=days, y=cumulative_biomass, mode='lines', name='Cumulative Biomass', line=dict(color='orange'))], layout=go.Layout(title="Cumulative Forecasted Biomass Over Time", xaxis={'title': "Days"}, yaxis={'title': "Cumulative Biomass (kg)"}))
    revenue_figure = go.Figure(data=[go.Scatter(x=days, y=cumulative_revenue_idr, mode='lines', name='Cumulative Revenue (IDR)', line=dict(color='purple'))], layout=go.Layout(title="Cumulative Forecasted Revenue Over Time (IDR)", xaxis={'title': "Days"}, yaxis={'title': "Cumulative Revenue (IDR)"}))

    revenue_output = f"Forecasted Revenue by Day {inputs['days_until_harvest']}: IDR {cumulative_revenue_idr[-1]:,.2f}"

    return sr_figure, abw_figure, biomass_figure, revenue_figure, revenue_output

if __name__ == '__main__':
    app.run_server(debug=True)
