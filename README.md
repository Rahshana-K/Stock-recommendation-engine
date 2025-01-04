# Stock Recommendation Engine

Unlock the secrets of the stock market with expert insights using this Stock Recommendation Engine! This project uses LSTM-based deep learning to predict stock prices and provide actionable buy/hold/sell recommendations based on historical data.



## Features

- *Stock Price Forecasting:* Predicts future stock prices using historical data.
  
- *Recommendation System:* Provides actionable recommendations (Buy, Hold, or Sell) based on forecasted price trends.
  
- *Visual Insights:* Interactive visualizations of stock prices and predictions.
  
- *Customizable Forecast Period:* Predict for the next day, next week, or next month.
  
- *Streamlit Integration:* Easy-to-use web interface for user interaction.



## Tech Stack

- **Programming Language:** Python
  
- **Libraries Used:**
  
  - **streamlit**: for creating the web app
  - **yfinance**: for fetching historical stock data
  - **numpy**,**pandas** : for data manipulation
  - **scikit-learn**: for data scaling
  - **tensorflow**,**keras**:for building the LSTM model
  - **matplotlib**: for data visualization


## Installation and Setup

1. Clone the repository:
  
   git clone https://github.com/Rahshana-K/stock-recommendation-engine.git
   

2. Navigate to the project directory:
   
   cd stock-recommendation-engine
   

3. Install the required dependencies:
  
   pip install -r requirements.txt
   

4. Run the Streamlit app:
  
   streamlit run app.py
   

5. Open the app in your browser at http://localhost:8501/.



## Usage

1. Select a stock ticker from the sidebar.
2. Choose the forecast duration (next day, week, or month).
3. View the stock price chart, including historical data and predictions.
4. Get actionable recommendations (Buy, Hold, or Sell).




## Example Screenshots

### Stock Price Visualization
![output1](https://github.com/user-attachments/assets/971c2d3c-ed97-4f4d-b383-71976a539736)

---
### Recommendation Output
![output2](https://github.com/user-attachments/assets/a2c12d76-7987-4d60-84ae-7470fec3f8ef)

---
### Selecting Tickers
![output3](https://github.com/user-attachments/assets/daf2f3d9-0235-4106-a4ed-43d7fd8f2ae0)


---
## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for review.


## Acknowledgments

- Yahoo Finance for stock data.
- TensorFlow and Keras for the LSTM model implementation.
- Streamlit for creating an intuitive web interface.

---
