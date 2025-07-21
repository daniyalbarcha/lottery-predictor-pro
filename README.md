# Lottery Predictor Pro

A sophisticated lottery number prediction tool built with Python and Streamlit.

## Features

- Web scraping of historical lottery data from multiple pages
- Advanced data analysis and pattern detection
- Machine Learning-based prediction using Random Forest
- AI model integration (GPT, Claude, Grok) for predictions
- Multiple prediction modes (single, reroll, 50x, 1000x)
- Top numbers analysis from multiple predictions
- Beautiful and intuitive Streamlit UI
- Secure API key management
- Sequential pattern detection
- Comprehensive statistical analysis

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Run the app: `streamlit run streamlit_app.py`
2. Choose data source (scrape new or upload existing)
3. View analysis in different tabs
4. Generate predictions using ML or AI models
5. Configure API keys in Settings page for AI models

## Requirements

See `requirements.txt` for full list of dependencies:
- streamlit
- pandas
- numpy
- scikit-learn
- beautifulsoup4
- requests
- plotly
- joblib
- tqdm

## Note

This tool is for educational purposes only. Past performance does not guarantee future results. 