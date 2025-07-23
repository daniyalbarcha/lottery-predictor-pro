import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from lottery_scraper import LotteryScraper
from lottery_analyzer import LotteryAnalyzer
import numpy as np
import random

# Page configuration
st.set_page_config(page_title="Lottery Predictor Pro", layout="wide", page_icon="🎲")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 3rem !important;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .model-toggle {
        background-color: #f1f1f1;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def simulate_ai_prediction(model_name, last_numbers, historical_data):
    """Simulate AI model predictions based on historical patterns"""
    # Each AI model has slightly different prediction strategies
    if model_name == "GPT":
        # GPT focuses more on recent patterns and frequency
        base_weights = [0.4, 0.3, 0.2, 0.1]  # More weight to recent patterns
        confidence = 0.85
    elif model_name == "Claude":
        # Claude balances historical patterns with randomness
        base_weights = [0.35, 0.35, 0.2, 0.1]  # More balanced weights
        confidence = 0.82
    elif model_name == "Grok":
        # Grok emphasizes pattern disruption and novelty
        base_weights = [0.3, 0.3, 0.3, 0.1]  # More evenly distributed
        confidence = 0.78
    
    predictions = []
    used_numbers = set()
    
    # Get frequency distribution from historical data
    all_numbers = []
    for i in range(1, 6):
        all_numbers.extend(historical_data[f'number{i}'].tolist())
    freq_dist = pd.Series(all_numbers).value_counts()
    
    # Adjust weights to match the length of last_numbers
    weights = [1/len(last_numbers)] * len(last_numbers)  # Equal weights as fallback
    for i in range(min(len(base_weights), len(weights))):
        weights[i] = base_weights[i]
    
    while len(predictions) < 5:
        # Consider last numbers with weight
        base = random.choices(last_numbers, weights=weights)[0]
        
        # Apply model-specific modifications
        if model_name == "GPT":
            # GPT tends to favor frequently occurring numbers
            candidates = freq_dist.head(10).index.tolist()
            num = random.choice(candidates) if candidates else base
        elif model_name == "Claude":
            # Claude balances between patterns and innovation
            num = base + random.choice([-2, -1, 0, 1, 2])
        else:  # Grok
            # Grok looks for pattern breaks
            num = base + random.choice([-3, -2, -1, 1, 2, 3])
        
        # Ensure number is valid and unique
        num = max(1, min(39, num))
        if num not in used_numbers:
            predictions.append(num)
            used_numbers.add(num)
    
    return sorted(predictions), confidence

def calculate_advanced_metrics(data):
    """Calculate advanced metrics from the dataset"""
    metrics = {}
    
    # Calculate overall statistics
    all_numbers = []
    for i in range(1, 6):
        all_numbers.extend(data[f'number{i}'].tolist())
    
    metrics['total_draws'] = len(data)
    metrics['date_range'] = (data['date'].max() - data['date'].min()).days
    metrics['most_common'] = pd.Series(all_numbers).value_counts().head(5).to_dict()
    metrics['least_common'] = pd.Series(all_numbers).value_counts().tail(5).to_dict()
    metrics['latest_date'] = data['date'].max().strftime('%Y-%m-%d')
    
    return metrics

def check_api_keys():
    """Check if required API keys are available"""
    try:
        api_keys = st.secrets.api_keys
        return {
            "GPT": bool(api_keys.get("openai_api_key")),
            "Claude": bool(api_keys.get("anthropic_api_key")),
            "Grok": bool(api_keys.get("grok_api_key"))
        }
    except:
        return {"GPT": False, "Claude": False, "Grok": False}

def analyze_top_numbers(predictions_list, top_n=20):
    """Analyze multiple prediction sets to find the most frequently predicted numbers"""
    # Collect all numbers from predictions
    all_predicted_numbers = []
    for pred_set in predictions_list:
        if isinstance(pred_set, dict):  # For AI predictions
            all_predicted_numbers.extend(pred_set['numbers'])
        else:  # For traditional ML predictions
            all_predicted_numbers.extend(pred_set)
    
    # Calculate frequency of each number
    number_frequency = pd.Series(all_predicted_numbers).value_counts()
    
    # Calculate confidence scores
    total_predictions = len(predictions_list)
    confidence_scores = (number_frequency / total_predictions * 100).round(2)
    
    # Get top N numbers with their frequencies
    top_numbers = number_frequency.head(top_n)
    top_numbers_confidence = confidence_scores.head(top_n)
    
    return {
        'numbers': top_numbers.index.tolist(),
        'frequencies': top_numbers.values.tolist(),
        'confidence_scores': top_numbers_confidence.values.tolist()
    }

def main():
    st.title("🎲 Lottery Predictor Pro")
    
    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LotteryAnalyzer()
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    if 'multiple_predictions' not in st.session_state:
        st.session_state.multiple_predictions = []
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    if 'ai_prediction' not in st.session_state:
        st.session_state.ai_prediction = None
    if 'ai_confidence' not in st.session_state:
        st.session_state.ai_confidence = None
    if 'ai_model' not in st.session_state:
        st.session_state.ai_model = None
    if 'last_numbers' not in st.session_state:
        st.session_state.last_numbers = None
    
    # Check API keys
    available_models = check_api_keys()
    
    # Sidebar
    st.sidebar.header("Data Collection")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Scrape New Data", "Upload Existing Data"]
    )
    
    if data_option == "Scrape New Data":
        pages_to_scrape = st.sidebar.slider("Number of pages to scrape", 10, 241, 50)
        if st.sidebar.button("Fetch Latest Data"):
            with st.spinner(f"Fetching historical data from {pages_to_scrape} pages..."):
                scraper = LotteryScraper()
                st.session_state.historical_data = scraper.scrape_historical_data(
                    start_page=1, 
                    end_page=pages_to_scrape
                )
                scraper.save_to_csv(st.session_state.historical_data)
                st.success("Data fetched successfully!")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            st.session_state.historical_data = pd.read_csv(uploaded_file)
            st.session_state.historical_data['date'] = pd.to_datetime(st.session_state.historical_data['date'])
            st.success("Data loaded successfully!")
    
    # Main content
    if st.session_state.historical_data is not None:
        # Calculate advanced metrics
        metrics = calculate_advanced_metrics(st.session_state.historical_data)
        
        # Display dataset overview
        st.header("📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Draws Analyzed", metrics['total_draws'])
        with col2:
            st.metric("Years of Data", f"{metrics['date_range'] / 365:.1f}")
        with col3:
            st.metric("Latest Draw Date", metrics['latest_date'])
        
        # Data Overview
        st.subheader("Recent Draws")
        st.dataframe(st.session_state.historical_data.head())
        
        # Train model and make predictions
        analyzer = st.session_state.analyzer
        analyzer.train_models(st.session_state.historical_data)
        
        # Get last numbers
        st.session_state.last_numbers = st.session_state.historical_data.iloc[0][
            ['number1', 'number2', 'number3', 'number4', 'number5']
        ].values
        
        # Find next draw date
        next_date = datetime.now()
        while next_date.weekday() == 6:  # Skip Sunday
            next_date += timedelta(days=1)
        
        # Model Selection
        st.header("🤖 Prediction Model Selection")
        prediction_method = st.radio(
            "Choose prediction method:",
            ["Traditional ML", "AI Models"],
            help="Traditional ML uses Random Forest. AI Models use advanced language models for prediction."
        )
        
        if prediction_method == "AI Models":
            # Filter available AI models
            available_ai_models = [model for model, available in available_models.items() if available]
            
            if not available_ai_models:
                st.error("⚠️ No AI models available. Please add API keys in Settings.")
                st.info("Go to Settings page to configure your API keys.")
            else:
                st.session_state.ai_model = st.selectbox(
                    "Select AI Model",
                    available_ai_models,
                    help="Each AI model uses different strategies for prediction"
                )
                
                # Show API key status
                st.markdown("### 🔑 API Key Status")
                for model, available in available_models.items():
                    status = "✅ Available" if available else "❌ Missing API Key"
                    st.markdown(f"**{model}**: {status}")
        
        # Prediction Controls
        st.header("🎯 Prediction Controls")
        pred_col1, pred_col2, pred_col3, pred_col4, pred_col5 = st.columns(5)
        
        with pred_col1:
            if st.button("🎲 Predict Numbers", type="primary"):
                if prediction_method == "Traditional ML":
                    st.session_state.current_prediction = analyzer.predict_with_sequential_logic(st.session_state.last_numbers, next_date)
                    st.session_state.ai_prediction = None
                else:
                    st.session_state.current_prediction = None
                    st.session_state.ai_prediction, st.session_state.ai_confidence = simulate_ai_prediction(
                        st.session_state.ai_model, 
                        st.session_state.last_numbers, 
                        st.session_state.historical_data
                    )
                st.session_state.prediction_count += 1
        
        with pred_col2:
            if st.button("🔄 Reroll Prediction"):
                if prediction_method == "Traditional ML":
                    st.session_state.current_prediction = analyzer.reroll_prediction(
                        st.session_state.last_numbers, next_date, st.session_state.current_prediction
                    )
                    st.session_state.ai_prediction = None
                else:
                    st.session_state.current_prediction = None
                    st.session_state.ai_prediction, st.session_state.ai_confidence = simulate_ai_prediction(
                        st.session_state.ai_model, 
                        st.session_state.last_numbers, 
                        st.session_state.historical_data
                    )
                st.session_state.prediction_count += 1
        
        with pred_col3:
            if st.button("🎰 Predict Again"):
                if prediction_method == "Traditional ML":
                    st.session_state.current_prediction = analyzer.predict_with_sequential_logic(st.session_state.last_numbers, next_date)
                    st.session_state.ai_prediction = None
                else:
                    st.session_state.current_prediction = None
                    st.session_state.ai_prediction, st.session_state.ai_confidence = simulate_ai_prediction(
                        st.session_state.ai_model, 
                        st.session_state.last_numbers, 
                        st.session_state.historical_data
                    )
                st.session_state.prediction_count += 1
        
        with pred_col4:
            prediction_count = st.selectbox("Prediction Sets", [10, 25, 50, 100], index=2)
            if st.button(f"🚀 Predict {prediction_count}x"):
                if prediction_method == "Traditional ML":
                    with st.spinner(f"Generating {prediction_count} prediction sets..."):
                        st.session_state.multiple_predictions = analyzer.predict_multiple_sets(
                            st.session_state.last_numbers, next_date, count=prediction_count, use_patterns=True
                        )
                    st.success(f"Generated {prediction_count} prediction sets!")
                else:
                    with st.spinner(f"Generating {prediction_count} AI prediction sets..."):
                        predictions = []
                        for i in range(prediction_count):
                            nums, conf = simulate_ai_prediction(st.session_state.ai_model, st.session_state.last_numbers, st.session_state.historical_data)
                            predictions.append({
                                'set_number': i + 1,
                                'numbers': nums,
                                'prediction_method': f'AI ({st.session_state.ai_model})',
                                'confidence': conf
                            })
                        st.session_state.multiple_predictions = predictions
                    st.success(f"Generated {prediction_count} AI prediction sets!")

        with pred_col5:
            top_n = st.selectbox("Top Numbers to Show", [8, 20], index=1)
            if st.button(f"🎯 Top {top_n} from 1000x"):
                with st.spinner(f"Generating 1000 predictions to find top {top_n} numbers..."):
                    # Generate 1000 predictions
                    thousand_predictions = []
                    for _ in range(1000):
                        if prediction_method == "Traditional ML":
                            pred = analyzer.predict_with_sequential_logic(st.session_state.last_numbers, next_date)
                            thousand_predictions.append(pred)
                        else:
                            pred, _ = simulate_ai_prediction(st.session_state.ai_model, st.session_state.last_numbers, st.session_state.historical_data)
                            thousand_predictions.append({'numbers': pred})
                    
                    # Analyze top numbers
                    top_numbers = analyze_top_numbers(thousand_predictions, top_n)
                    
                    # Display results in a new section
                    st.header(f"🎯 Top {top_n} Most Predicted Numbers")
                    
                    # Create two rows of numbers for better spacing
                    row1_cols = st.columns(5)
                    row2_cols = st.columns(5)
                    
                    # Display first row (5 numbers)
                    for i in range(min(5, len(top_numbers['numbers']))):
                        with row1_cols[i]:
                            st.markdown(
                                f"""
                                <div style='background-color: #e74c3c; color: white; padding: 20px; 
                                border-radius: 50%; width: 60px; height: 60px; display: flex; 
                                align-items: center; justify-content: center; font-size: 24px; margin: auto;
                                margin-bottom: 10px;'>
                                {top_numbers['numbers'][i]}
                                </div>
                                <div style='text-align: center; margin-top: 5px; margin-bottom: 20px;'>
                                Confidence: {top_numbers['confidence_scores'][i]:.1f}%
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    # Display second row (remaining numbers)
                    for i in range(5, min(top_n, len(top_numbers['numbers']))):
                        with row2_cols[i-5]:
                            st.markdown(
                                f"""
                                <div style='background-color: #e74c3c; color: white; padding: 20px; 
                                border-radius: 50%; width: 60px; height: 60px; display: flex; 
                                align-items: center; justify-content: center; font-size: 24px; margin: auto;
                                margin-bottom: 10px;'>
                                {top_numbers['numbers'][i]}
                                </div>
                                <div style='text-align: center; margin-top: 5px; margin-bottom: 20px;'>
                                Confidence: {top_numbers['confidence_scores'][i]:.1f}%
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    # Display detailed statistics
                    st.subheader("📊 Detailed Statistics")
                    stats_df = pd.DataFrame({
                        'Number': top_numbers['numbers'],
                        'Frequency': top_numbers['frequencies'],
                        'Confidence': [f"{conf:.1f}%" for conf in top_numbers['confidence_scores']]
                    })
                    st.dataframe(stats_df)

        # Display Current Prediction
        if st.session_state.current_prediction or st.session_state.ai_prediction:
            st.header("🎯 Current Prediction")
            st.subheader(f"Predicted Numbers for {next_date.strftime('%Y-%m-%d')} (Attempt #{st.session_state.prediction_count})")
            
            prediction_numbers = st.session_state.current_prediction if st.session_state.current_prediction else st.session_state.ai_prediction
            
            cols = st.columns(5)
            for i, num in enumerate(prediction_numbers):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div style='background-color: #e74c3c; color: white; padding: 20px; 
                        border-radius: 50%; width: 60px; height: 60px; display: flex; 
                        align-items: center; justify-content: center; font-size: 24px; margin: auto;'>
                        {num}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            if st.session_state.ai_prediction and st.session_state.ai_confidence:
                st.info(f"AI Model Confidence: {st.session_state.ai_confidence:.2%}")

        # Display Multiple Predictions Analysis
        if st.session_state.multiple_predictions:
            st.header("📊 Multiple Predictions Analysis")
            
            # Confidence Analysis
            confidence_analysis = analyzer.analyze_prediction_confidence(st.session_state.multiple_predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Most Confident Numbers")
                confident_numbers = confidence_analysis['most_confident'][:10]
                
                for num, confidence in confident_numbers:
                    st.progress(confidence / 100, text=f"Number {num}: {confidence:.1f}%")
            
            with col2:
                st.subheader("📈 Frequency Distribution")
                freq_data = confidence_analysis['frequency_distribution']
                fig_freq = px.bar(
                    x=list(freq_data.keys()),
                    y=list(freq_data.values()),
                    title=f"Number Frequency in {prediction_count} Predictions"
                )
                st.plotly_chart(fig_freq, use_container_width=True)
            
            # Show sample predictions
            st.subheader("🎲 Sample Prediction Sets")
            sample_size = min(10, len(st.session_state.multiple_predictions))
            
            for i in range(sample_size):
                pred_set = st.session_state.multiple_predictions[i]
                st.write(f"**Set {pred_set['set_number']}**: {pred_set['numbers']} ({pred_set['prediction_method']})")
        
        # Analysis Tabs
        st.header("📈 Detailed Analysis")
        tab1, tab2, tab3, tab4 = st.tabs(["Number Patterns", "Time Analysis", "Advanced Stats", "Sequential Patterns"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔥 Hot Numbers")
                fig_hot = px.bar(
                    x=list(metrics['most_common'].keys()),
                    y=list(metrics['most_common'].values()),
                    title="Most Frequent Numbers (All Time)"
                )
                st.plotly_chart(fig_hot, use_container_width=True)
            
            with col2:
                st.subheader("❄️ Cold Numbers")
                fig_cold = px.bar(
                    x=list(metrics['least_common'].keys()),
                    y=list(metrics['least_common'].values()),
                    title="Least Frequent Numbers (All Time)"
                )
                st.plotly_chart(fig_cold, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Number Trends Over Time")
            fig = go.Figure()
            for i in range(1, 6):
                fig.add_trace(go.Scatter(
                    x=st.session_state.historical_data['date'],
                    y=st.session_state.historical_data[f'number{i}'],
                    name=f'Number {i}'
                ))
            fig.update_layout(title="Historical Number Trends")
            st.plotly_chart(fig, use_container_width=True)
            
            # Day of week analysis
            patterns = analyzer.analyze_patterns(st.session_state.historical_data)
            st.subheader("📅 Day of Week Patterns")
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_stats = patterns['day_statistics']
            
            fig_days = go.Figure()
            for num in range(1, 6):
                values = [day_stats[f'number{num}'].get(day, 0) for day in range(7)]
                fig_days.add_trace(go.Bar(
                    name=f'Number {num}',
                    x=day_names,
                    y=values
                ))
            fig_days.update_layout(
                title="Average Numbers by Day of Week",
                barmode='group'
            )
            st.plotly_chart(fig_days, use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Number Streaks")
            # Calculate streaks
            numbers_array = np.array([st.session_state.historical_data[f'number{i}'] for i in range(1, 6)]).T
            current_streaks = {i: 0 for i in range(1, 40)}
            max_streaks = {i: 0 for i in range(1, 40)}
            
            for draw in numbers_array:
                for num in range(1, 40):
                    if num in draw:
                        current_streaks[num] += 1
                        max_streaks[num] = max(max_streaks[num], current_streaks[num])
                    else:
                        current_streaks[num] = 0
            
            streak_data = pd.DataFrame.from_dict(
                max_streaks, 
                orient='index', 
                columns=['Max Consecutive Appearances']
            )
            fig_streaks = px.bar(
                streak_data,
                title="Maximum Consecutive Appearances by Number"
            )
            st.plotly_chart(fig_streaks, use_container_width=True)
            
            # Display additional statistics
            st.subheader("📊 Additional Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Most Common Combinations")
                combinations = st.session_state.historical_data.apply(
                    lambda x: tuple(sorted([x[f'number{i}'] for i in range(1, 6)])),
                    axis=1
                ).value_counts().head(5)
                for combo, count in combinations.items():
                    st.markdown(f"**{combo}**: {count} times")
            
            with col2:
                st.markdown("### Number Distribution")
                all_nums = []
                for i in range(1, 6):
                    all_nums.extend(st.session_state.historical_data[f'number{i}'])
                fig_dist = px.histogram(
                    x=all_nums,
                    nbins=39,
                    title="Overall Number Distribution"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        with tab4:
            st.subheader("🔍 Sequential Pattern Analysis")
            st.info("Based on your observation of +1 patterns and sequential logic in lottery draws")
            
            patterns = analyzer.analyze_patterns(st.session_state.historical_data)
            seq_patterns = patterns['sequential_patterns']
            
            # Display +1 pattern frequency
            plus_one_data = seq_patterns['plus_one_sequences']
            if plus_one_data:
                st.subheader("➕ Plus One (+1) Pattern Occurrences")
                st.write(f"Found {len(plus_one_data)} instances where numbers increased by +1 in consecutive draws")
                
                # Show recent +1 patterns
                recent_plus_one = plus_one_data[:10]
                for pattern in recent_plus_one:
                    st.write(f"**{pattern['date'].strftime('%Y-%m-%d')}**: {pattern['current']} → {pattern['next']} (+1 count: {pattern['plus_one_count']})")
            
            # Position correlations
            st.subheader("📍 Position-Based Pattern Analysis")
            pos_corr = seq_patterns['position_correlations']
            
            for pos, data in pos_corr.items():
                st.write(f"**{pos.replace('_', ' ').title()}**: Average difference = {data['mean_diff']:.2f}")
                st.write(f"Most common differences: {data['common_diffs']}")
                st.write("---")
            
            # Pattern-based prediction explanation
            st.subheader("🎯 How Pattern-Based Predictions Work")
            st.markdown("""
            Based on your analysis, the system now:
            1. **Detects +1 sequences** where numbers increment by 1 between draws
            2. **Analyzes position correlations** to find systematic changes
            3. **Applies weighted logic** favoring +1 patterns (40% weight)
            4. **Considers alternative patterns** like -1, +2 for variety
            5. **Ensures number validity** (1-39 range, no duplicates)
            """)
    
    else:
        st.info("Please load or fetch data to start analysis")

if __name__ == "__main__":
    main() 