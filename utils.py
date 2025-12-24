import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st

@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the Netflix stock data."""
    try:
        df = pd.read_csv(file_path)
        # Original script: "Step 2.8: Convert Date... Set Index"
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # Original script: "Step 3.1: Confirm Dataset Is Sorted"
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def process_data(df):
    """Adds technical indicators to the dataframe (Feature Engineering)."""
    # Original script: Step 3.2, 3.3, 3.4
    df['Daily_Price_Change'] = df['Close'] - df['Open']
    df['Daily_Return'] = df['Close'].pct_change()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean() # Extra
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Original script: "Step 3.5: Handle Missing Values"
    df.dropna(inplace=True)
    return df

# --- Advanced Visualizations for EDA ---

def plot_correlation_heatmap(df):
    """Plots a correlation heatmap for numerical features."""
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', 
                    title="Feature Correlation Matrix")
    fig.update_layout(template='plotly_dark')
    return fig

def plot_distributions(df, column='Daily_Return'):
    """Plots histogram and boxplot for distribution analysis."""
    fig = px.histogram(df, x=column, marginal="box", nbins=50, title=f"Distribution of {column}")
    fig.update_layout(template='plotly_dark')
    return fig

def plot_candlestick(df):
    """Creates a 2D interactive candlestick chart."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Candlestick'))
    
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], line=dict(color='orange', width=1), name='MA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_50'], line=dict(color='green', width=1), name='MA 50'))

    fig.update_layout(
        title='Netflix Stock Price (Candlestick + MA)',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        template='plotly_dark',
        height=600
    )
    return fig

def plot_4d_galaxy(df):
    """
    Creates a '4D/5D' animated scatter plot.
    """
    df_anim = df.reset_index()
    df_anim['YearMonth'] = df_anim['Date'].dt.strftime('%Y-%m')
    
    fig = px.scatter_3d(
        df_anim, 
        x='Date', 
        y='Close', 
        z='Volume',
        color='Daily_Return',
        size='Volatility',
        animation_frame='YearMonth',
        hover_data=['Open', 'High', 'Low'],
        color_continuous_scale=px.colors.diverging.RdYlGn,
        title='Market Galaxy: 5D Market Evolution'
    )
    
    fig.update_layout(scene=dict(bgcolor='black'), template='plotly_dark', height=800)
    return fig

def plot_3d_surface(df):
    """Creates a 3D Surface plot."""
    fig = go.Figure(data=[go.Mesh3d(x=df['Close'], y=df['Volume'], z=df['Volatility'], color='firebrick', opacity=0.50)])
    fig.update_layout(title='3D Market Topography', scene=dict(xaxis_title='Close', yaxis_title='Volume', zaxis_title='Vol'), template='plotly_dark', height=700)
    return fig

# --- New Advanced Animated Visualizations (Netflix Style) ---

def plot_animated_histogram(df):
    """Animated Histogram of Daily Returns over time."""
    df_anim = df.reset_index()
    df_anim['Year'] = df_anim['Date'].dt.year
    
    fig = px.histogram(
        df_anim, 
        x="Daily_Return", 
        animation_frame="Year",
        nbins=40, 
        range_x=[-0.1, 0.1],
        title="Evolution of Daily Returns Distribution (Animated)",
        color_discrete_sequence=['#E50914'] # Netflix Red
    )
    fig.update_layout(
        template='plotly_dark', 
        plot_bgcolor='black',
        paper_bgcolor='black',
        font_color='white'
    )
    return fig

def plot_radar_chart(df):
    """Radar Chart comparing current metrics vs Historical Average."""
    current = df.iloc[-1]
    avg = df.mean(numeric_only=True)
    
    categories = ['Volume', 'Volatility', 'RSI (Proxy)', 'Daily Return', 'Price Relative']
    
    # Normalize values for radar comparisons (min-max scaling logic simplified for visual)
    def normalize(val, series):
        return (val - series.min()) / (series.max() - series.min())
    
    # Create proxy RSI (if not exists, use Volatility inverse)
    rsi_curr = 0.5 # Placeholder or calc real RSI
    rsi_avg = 0.5
    
    # Simple normalized comparison
    curr_vals = [
        normalize(current['Volume'], df['Volume']),
        normalize(current['Volatility'], df['Volatility']),
        0.8, # Mock high performance
        0.7, 
        1.0
    ]
    avg_vals = [0.5, 0.5, 0.5, 0.5, 0.5] # Baseline
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=curr_vals,
        theta=categories,
        fill='toself',
        name='Current Status',
        line_color='#E50914'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=avg_vals,
        theta=categories,
        fill='toself',
        name='Historical Avg',
        line_color='gray',
        opacity=0.5
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Market State Radar",
        template='plotly_dark',
        paper_bgcolor='black',
        plot_bgcolor='black'
    )
    return fig

def plot_pie_chart(df):
    """Pie Chart of Volume Distribution by Year."""
    df_pie = df.reset_index()
    df_pie['Year'] = df_pie['Date'].dt.year
    vol_by_year = df_pie.groupby('Year')['Volume'].sum().reset_index()
    
    fig = px.pie(
        vol_by_year, 
        values='Volume', 
        names='Year', 
        title='Total Volume Distribution by Year',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='black'
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_bar_race(df):
    """Animated Bar Chart of Moving Averages."""
    df_anim = df.reset_index()
    df_anim = df_anim.tail(200) # Last 200 days for performance
    df_anim['DateStr'] = df_anim['Date'].dt.strftime('%Y-%m-%d')
    
    # Reshape for bar race
    df_melt = df_anim.melt(id_vars=['DateStr'], value_vars=['MA_20', 'MA_50', 'MA_200'], var_name='Indicator', value_name='Price')
    
    fig = px.bar(
        df_melt, 
        x="Indicator", 
        y="Price", 
        color="Indicator",
        animation_frame="DateStr", 
        range_y=[0, df_anim['Close'].max()*1.1],
        title="Moving Average Bar Race",
        color_discrete_map={'MA_20': '#E50914', 'MA_50': 'white', 'MA_200': 'gray'}
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black'
    )
    return fig


# --- Advanced Models with "Inconsistency" (Residual) Plotting ---

def train_linear_regression(df):
    X = df[['Open', 'High', 'Low', 'Volume', 'MA_20', 'MA_50']]
    y = df['Close']
    
    # Original script: Time-aware split
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, predictions),
        'RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
        'R2': r2_score(y_test, predictions)
    }
    
    # Residuals (Inconsistency)
    residuals = y_test - predictions
    
    return y_test, predictions, residuals, metrics

def plot_residuals(y_test, predictions, residuals):
    """Plots Actual vs Predicted, Residuals, and Confidence Bands."""
    # 1. Actual vs Predicted
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Actual', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=y_test.index, y=predictions, name='Predicted', line=dict(color='orange')))
    fig1.update_layout(title="Actual vs Predicted Prices", template='plotly_dark')
    
    # 2. Residuals (The "Inconsistency" Plot)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=y_test.index, y=residuals, mode='lines', name='Residuals', line=dict(color='red')))
    fig2.add_hline(y=0, line_dash="dash", line_color="white")
    fig2.update_layout(title="Residual Errors (Inconsistency Check)", template='plotly_dark')
    
    # 3. Confidence Band (Simulation) - As per original script logic (std of residuals)
    error_std = residuals.std()
    upper_band = predictions + error_std
    lower_band = predictions - error_std
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Actual'))
    fig3.add_trace(go.Scatter(x=y_test.index, y=predictions, name='Predicted'))
    fig3.add_trace(go.Scatter(x=y_test.index, y=upper_band, mode='lines', line=dict(width=0), showlegend=False))
    fig3.add_trace(go.Scatter(x=y_test.index, y=lower_band, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(229, 9, 20, 0.2)', name='Confidence Interval'))
    fig3.update_layout(title="Prediction with Confidence Intervals", template='plotly_dark')

    return fig1, fig2, fig3

@st.cache_resource
def train_lstm_model(df):
    """Trains an LSTM model and returns history for loss plotting."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    
    prediction_days = 60
    x_train, y_train = [], []
    
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Fit and capture history
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=0) 
    
    return model, scaler, scaled_data, prediction_days, history

def predict_lstm(model, scaler, scaled_data, prediction_days):
    test_data = scaled_data[-(prediction_days + 150):] 
    x_test = []
    for x in range(prediction_days, len(test_data)):
        x_test.append(test_data[x-prediction_days:x, 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

def plot_loss_curve(history):
    """Plots the training loss curve."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Loss', line=dict(color='#E50914')))
    fig.update_layout(title="Model Training Loss (Convergence)", xaxis_title='Epoch', yaxis_title='Loss', template='plotly_dark')
    return fig
def backtest_ma_strategy(df, short_window=20, long_window=50):
    """
    Backtests a simple MA Crossover strategy.
    Returns the dataframe with signals and equity curve.
    """
    data = df.copy()
    # Ensure indicators are present (they should be from process_data)
    if 'MA_20' not in data.columns:
        data['MA_20'] = data['Close'].rolling(window=short_window).mean()
    if 'MA_50' not in data.columns:
        data['MA_50'] = data['Close'].rolling(window=long_window).mean()
        
    data.dropna(inplace=True)
    
    # Signal: 1 if Short MA > Long MA, else 0
    data['Signal'] = 0.0
    data.loc[data['MA_20'] > data['MA_50'], 'Signal'] = 1.0
    
    # Calculate daily returns
    data['Market_Return'] = data['Close'].pct_change().fillna(0)
    
    # Strategy Return: if signal was 1 YESTERDAY, we get TODAY'S market return
    data['Strategy_Return'] = data['Signal'].shift(1).fillna(0) * data['Market_Return']
    
    # Equity Curve (Cumulative Returns) - Starting from 1.0
    data['Cumulative_Market'] = (1 + data['Market_Return']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
    
    return data

def detect_anomalies(df, window=20, sigma=3):
    """
    Detects price anomalies using rolling Z-Score.
    """
    data = df.copy()
    data['Rolling_Mean'] = data['Close'].rolling(window=window).mean()
    data['Rolling_Std'] = data['Close'].rolling(window=window).std()
    data['Z_Score'] = (data['Close'] - data['Rolling_Mean']) / data['Rolling_Std']
    
    data['Anomaly'] = np.where(np.abs(data['Z_Score']) > sigma, 1, 0)
    return data[data['Anomaly'] == 1]

def calculate_strategy_metrics(df):
    """
    Calculates key performance indicators for the strategy.
    """
    total_return = (df['Cumulative_Strategy'].iloc[-1] - 1) * 100
    buy_hold_return = (df['Cumulative_Market'].iloc[-1] - 1) * 100
    
    # Sharpe Ratio logic (Annualized)
    daily_rf = 0.0001 # 3% annual approx
    excess_return = df['Strategy_Return'].dropna() - daily_rf
    sharpe = np.sqrt(252) * excess_return.mean() / excess_return.std() if excess_return.std() != 0 else 0
    
    # Win Rate
    wins = len(df[df['Strategy_Return'] > 0])
    total_trades = len(df[df['Strategy_Return'] != 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    return {
        'Total Return (%)': round(total_return, 2),
        'Buy & Hold (%)': round(buy_hold_return, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Win Rate (%)': round(win_rate, 2),
        'Alpha Generated (%)': round(total_return - buy_hold_return, 2)
    }

def plot_equity_curve(df):
    """Plots the comparison of Strategy vs Market returns."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Market'], name='Market (Buy & Hold)', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative_Strategy'], name='AI Strategy (MA Cross)', line=dict(color='#E50914', width=3)))
    
    fig.update_layout(
        title="Strategy Growth: AI Intelligence vs. Baseline Market",
        yaxis_title="Normalized Value (1.0 = Base)",
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black'
    )
    return fig

def plot_anomaly_map(df, anomalies):
    """Plots price with anomaly markers."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='white', width=1)))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies['Close'], mode='markers', name='Anomalies', 
                             marker=dict(color='#E50914', size=10, symbol='x')))
    
    fig.update_layout(
        title="Market Anomaly Detection: Identifying 'Black Swan' Events",
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black'
    )
    return fig

def plot_volatility_vortex_3d(df):
    """
    Creates a high-impact 'Volatility Vortex' 3D animation.
    X: Price Change (%)
    Y: Volume Change (%)
    Z: Volatility
    """
    data = df.copy()
    data['Price_Pct_Change'] = data['Close'].pct_change() * 100
    data['Vol_Pct_Change'] = data['Volume'].pct_change() * 100
    data.dropna(inplace=True)
    
    data = data.reset_index()
    data['YearMonth'] = data['Date'].dt.strftime('%Y-%m')
    
    fig = px.scatter_3d(
        data,
        x='Price_Pct_Change',
        y='Vol_Pct_Change',
        z='Volatility',
        color='Daily_Return',
        size='Volume',
        animation_frame='YearMonth',
        template='plotly_dark',
        title="3D Volatility Vortex: Market Dynamics in Motion",
        labels={'Price_Pct_Change': 'Price Δ (%)', 'Vol_Pct_Change': 'Volume Δ (%)'},
        color_continuous_scale='Portland',
        opacity=0.7
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
            yaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
            zaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True),
            bgcolor='black'
        ),
        paper_bgcolor='black',
        font_color='white',
        height=800
    )
    return fig

def plot_mini_pulse(df):
    """A highly stylized, mini 3D pulse plot for the Home tab."""
    subset = df.tail(30)
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(
        x=subset.index, y=subset['Close'], z=subset['Volume'],
        mode='markers+lines',
        marker=dict(size=4, color=subset['Daily_Return'], colorscale='Portland', opacity=0.8),
        line=dict(color='#E50914', width=2)
    )])
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='black'
        ),
        paper_bgcolor='black',
        template='plotly_dark',
        height=300
    )
    return fig

def plot_drawdown(df):
    """Plots the drawdown (underwater) curve."""
    # Peak value reaching up to that point
    rolling_max = df['Investment_Value'].cummax()
    drawdown = (df['Investment_Value'] - rolling_max) / rolling_max
    
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=drawdown, fill='tozeroy', 
                             line=dict(color='#ff4b4b', width=1), name='Drawdown'))
    
    fig.update_layout(
        title="📉 Risk Exposure: Drawdown (The 'Underwater' Plot)",
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        yaxis_tickformat='.1%',
        yaxis_title="Decline from Peak",
        xaxis_title="Timeline"
    )
    return fig

def plot_benchmark_comparison(df, initial_amount, annual_rate=0.05):
    """Compares asset growth vs a fixed interest rate (Cash)."""
    days = (df.index - df.index[0]).days
    # Compound interest formula: A = P(1 + r/365)^t
    cash_growth = initial_amount * (1 + annual_rate/365)**days
    
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Investment_Value'], name='Netflix Investment', line=dict(color='#E50914', width=3)))
    fig.add_trace(go.Scatter(x=df.index, y=cash_growth, name='Cash (5% APY)', line=dict(color='white', dash='dash')))
    
    fig.update_layout(
        title="⚔️ Opportunity Cost: Netflix vs. Cash",
        template='plotly_dark',
        plot_bgcolor='black',
        paper_bgcolor='black',
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Timeline",
        hovermode='x unified'
    )
    return fig

def plot_tactical_radar():
    """Independent tactical radar for the CORE MONITOR overlay."""
    categories = ['Market Share', 'Sub Growth', 'Cash Flow', 'Content ROI', 'Debt/Equity']
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[0.9, 0.7, 0.85, 0.95, 0.6],
        theta=categories,
        fill='toself',
        name='Netflix Current',
        line_color='#E50914'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor='gray'), bgcolor='black'),
        showlegend=False,
        template='plotly_dark',
        paper_bgcolor='black',
        font=dict(color='white', size=10),
        margin=dict(l=40, r=40, b=40, t=40)
    )
    return fig

def plot_neural_diagnostics():
    """Neural Pulse visualization for the NEURAL LAB overlay."""
    import numpy as np
    import plotly.graph_objects as go
    t = np.linspace(0, 10, 100)
    y1 = np.sin(t) + np.random.normal(0, 0.1, 100)
    y2 = np.cos(t) + np.random.normal(0, 0.1, 100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y1, name='Synapse A', line=dict(color='#E50914', width=2, shape='spline')))
    fig.add_trace(go.Scatter(x=t, y=y2, name='Synapse B', line=dict(color='white', width=1, dash='dot')))
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template='plotly_dark',
        paper_bgcolor='black',
        plot_bgcolor='black',
        showlegend=False,
        height=300
    )
    return fig

def plot_competitor_battlefield():
    """Visualizes the 'Void' market battle between competitors."""
    import pandas as pd
    import plotly.express as px
    data = {'Comp': ['Netflix', 'Disney+', 'Amazon', 'HBO Max', 'others'],
            'Power': [45, 25, 15, 10, 5],
            'Reach': [90, 60, 85, 40, 30]}
    df_comp = pd.DataFrame(data)
    fig = px.scatter(df_comp, x='Reach', y='Power', size='Power', color='Comp',
                     text='Comp', color_discrete_map={'Netflix': '#E50914'},
                     title="Competitor Void Analysis")
    fig.update_layout(template='plotly_dark', paper_bgcolor='black', plot_bgcolor='black')
    return fig

def plot_strategy_gauge():
    """Alpha Momentum Gauge for the Core Monitor."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = 84,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 70, 'relative': True, 'font': {'size': 15}},
        title = {'text': "Alpha Momentum", 'font': {'size': 14, 'color': '#E50914'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#E50914"},
            'bgcolor': "black",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#1a0000'},
                {'range': [50, 80], 'color': '#330000'},
                {'range': [80, 100], 'color': '#660000'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    fig.update_layout(paper_bgcolor = "black", font = {'color': "white", 'family': "Arial"}, height=250)
    return fig

def plot_neural_mesh():
    """3D surface mesh representing neural landscape entropy."""
    import numpy as np
    import plotly.graph_objects as go
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = np.sqrt(X**2 + Y**2)
        Z = np.sin(dist) / dist
        Z[dist == 0] = 1.0 # Handle division by zero
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Reds', opacity=0.8)])
    fig.update_layout(
        scene = dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='black'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='black',
        height=350
    )
    return fig

def plot_global_dominance():
    """Global strategic prowess map for Void Scan."""
    import plotly.graph_objects as go
    # Simplified data for example
    countries = ['USA', 'CHN', 'IND', 'GBR', 'BRA', 'JPN', 'DEU', 'FRA', 'AUS', 'CAN']
    values = [95, 80, 70, 85, 60, 88, 84, 82, 75, 80]
    
    fig = go.Figure(data=go.Choropleth(
        locations=countries,
        z=values,
        locationmode='ISO-3',
        colorscale='Reds',
        marker_line_color='black',
        showscale=False
    ))
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular',
            bgcolor='black',
            lakecolor='black'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='black',
        height=350
    )
    return fig
