import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import datetime
import os
from typing import Optional, Dict, Any, List
# filepath: e:\earnings_agent\app.py
import os
print(f"FMP_API_KEY: {os.getenv('FMP_API_KEY')}")  # Add this line for debugging
from src.fetch.schedule import get_upcoming_earnings
from src.fetch.transcript import get_transcript
from src.analyze.metrics import analyze_transcript_metrics
from src.analyze.sentiment import analyze_transcript_sentiment, generate_sentiment_summary
from src.analyze.comparison import analyze_quarterly_comparison, generate_comparison_summary
from src.generate.summary import generate_comprehensive_summary, generate_markdown_summary
from src.generate.pdf import generate_pdf_report
from src.utils import log_message, format_currency, format_percentage, slugify
from config import PDF_OUTPUT_DIR

# Page configuration
st.set_page_config(
    page_title="Earnings Call Intelligence Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
    .highlight-positive {
        color: #4CAF50;
    }
    .highlight-negative {
        color: #F44336;
    }
    .highlight-neutral {
        color: #FFC107;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = ""
if 'current_quarter' not in st.session_state:
    st.session_state.current_quarter = None
if 'current_year' not in st.session_state:
    st.session_state.current_year = None
if 'transcript_data' not in st.session_state:
    st.session_state.transcript_data = None
if 'metrics_data' not in st.session_state:
    st.session_state.metrics_data = None
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = None
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
if 'summary_data' not in st.session_state:
    st.session_state.summary_data = None

def fetch_data(symbol: str, quarter: Optional[int], year: Optional[int]) -> None:
    """Fetch all relevant data for a given symbol, quarter and year."""
    with st.spinner(f"Fetching transcript for {symbol} Q{quarter} {year}..."):
        st.session_state.transcript_data = get_transcript(symbol, quarter, year)
    
    if st.session_state.transcript_data.get("error"):
        st.error(f"Error fetching transcript: {st.session_state.transcript_data['error']}")
        return False
    
    with st.spinner("Analyzing financial metrics..."):
        st.session_state.metrics_data = analyze_transcript_metrics(symbol, quarter, year)
    
    with st.spinner("Analyzing sentiment..."):
        st.session_state.sentiment_data = analyze_transcript_sentiment(symbol, quarter, year)
    
    with st.spinner("Comparing with previous quarter..."):
        st.session_state.comparison_data = analyze_quarterly_comparison(symbol, quarter, year)
    
    with st.spinner("Generating comprehensive summary..."):
        st.session_state.summary_data = generate_comprehensive_summary(symbol, quarter, year)
    
    st.session_state.current_symbol = symbol
    st.session_state.current_quarter = quarter
    st.session_state.current_year = year
    
    return True

def display_metrics_card(metrics: Dict[str, Any]) -> None:
    """Display a card with metrics data."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Key Financial Metrics</h3>', unsafe_allow_html=True)
    
    # Create columns for metrics
    cols = st.columns(3)
    
    # Get metrics data
    financial_metrics = metrics.get("metrics", {})
    
    # Display metrics
    if financial_metrics:
        for i, (key, value) in enumerate(financial_metrics.items()):
            with cols[i % 3]:
                if isinstance(value, (int, float)):
                    if "revenue" in key.lower() or "income" in key.lower() or "profit" in key.lower():
                        formatted_value = format_currency(value)
                    elif "margin" in key.lower() or "growth" in key.lower() or "rate" in key.lower():
                        formatted_value = format_percentage(value)
                    else:
                        formatted_value = str(value)
                else:
                    formatted_value = str(value)
                
                st.markdown(f'<div class="metric-value">{formatted_value}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">{key}</div>', unsafe_allow_html=True)
    else:
        st.warning("No financial metrics available.")
    
    # Display guidance
    st.markdown('<h3 class="sub-header">Forward Guidance</h3>', unsafe_allow_html=True)
    guidance = metrics.get("guidance", {})
    
    if guidance:
        for key, value in guidance.items():
            st.markdown(f"**{key}**: {value}")
    else:
        st.info("No forward guidance provided.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_sentiment_chart(sentiment: Dict[str, Any]) -> None:
    """Display sentiment analysis charts."""
    if not sentiment or "speaker_sentiment" not in sentiment:
        st.warning("No sentiment data available")
        return

    # Create sentiment trend chart
    sentiment_scores = sentiment.get("sentiment_over_time", [])
    if sentiment_scores:
        df = pd.DataFrame(sentiment_scores)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['sentiment_score'],
            mode='lines+markers',
            name='Sentiment Score'
        ))

        fig.update_layout(
            title="Sentiment Trend During Call",
            xaxis_title="Time",
            yaxis_title="Sentiment Score",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Create speaker sentiment chart
    speaker_sentiment = sentiment.get("speaker_sentiment", {})
    if speaker_sentiment:
        # Convert dictionary to DataFrame with explicit types
        speakers_data = []
        for speaker, sentiment_info in speaker_sentiment.items():
            # Check if sentiment_info is a dictionary
            if isinstance(sentiment_info, dict) and "sentiment_score" in sentiment_info:
                score = sentiment_info["sentiment_score"]
            else:
                score = 0.0  # Default score if not found

            speakers_data.append({"Speaker": str(speaker), "Sentiment": float(score)})

        speakers_df = pd.DataFrame(speakers_data)

        # Create bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=speakers_df['Speaker'],
            y=speakers_df['Sentiment'],
            marker_color=speakers_df['Sentiment'],
            marker=dict(
                colorscale=[[0, 'red'], [0.5, 'yellow'], [1.0, 'green']],
                cmin=-1,
                cmax=1
            )
        ))

        fig.update_layout(
            title="Sentiment by Speaker",
            xaxis_title="Speaker",
            yaxis_title="Sentiment Score",
            yaxis=dict(range=[-1, 1])
        )

        st.plotly_chart(fig, use_container_width=True)

def display_comparison(comparison: Dict[str, Any]) -> None:
    """Display quarterly comparison analysis."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Quarter-over-Quarter Comparison</h3>', unsafe_allow_html=True)
    
    metrics_comparison = comparison.get("metrics_comparison", {})
    
    if metrics_comparison:
        # Create a table for comparison
        columns = ["Metric", "Current Quarter", "Previous Quarter", "Change", "% Change"]
        data = []
        
        for metric, details in metrics_comparison.items():
            current = details.get("current")
            previous = details.get("previous")
            change = details.get("change")
            percent_change = details.get("percent_change")
            
            if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
                if "revenue" in metric.lower() or "income" in metric.lower() or "profit" in metric.lower():
                    current_formatted = format_currency(current)
                    previous_formatted = format_currency(previous)
                    change_formatted = format_currency(change)
                elif "margin" in metric.lower() or "growth" in metric.lower() or "rate" in metric.lower():
                    current_formatted = format_percentage(current)
                    previous_formatted = format_percentage(previous)
                    change_formatted = format_percentage(change)
                else:
                    current_formatted = str(current)
                    previous_formatted = str(previous)
                    change_formatted = str(change)
                
                percent_change_formatted = f"{percent_change:.2f}%"
            else:
                current_formatted = str(current)
                previous_formatted = str(previous)
                change_formatted = "N/A"
                percent_change_formatted = "N/A"
            
            data.append([metric, current_formatted, previous_formatted, change_formatted, percent_change_formatted])
        
        df = pd.DataFrame(data, columns=columns)
        st.dataframe(df, use_container_width=True)
        
        # Display comparison summary
        summary = generate_comparison_summary(comparison)
        st.markdown(f"**Summary**: {summary}")
    else:
        st.warning("No comparison data available.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_transcript_summary(transcript: Dict[str, Any]) -> None:
    """Display transcript summary."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Transcript Overview</h3>', unsafe_allow_html=True)
    
    title = transcript.get("title", "Unknown")
    date = transcript.get("date", "Unknown")
    company = transcript.get("company", "Unknown")
    
    st.markdown(f"**Company**: {company}")
    st.markdown(f"**Title**: {title}")
    st.markdown(f"**Date**: {date}")
    
    # Display participants if available
    participants = transcript.get("participants", [])
    if participants:
        st.markdown("### Participants")
        cols = st.columns(2)
        
        company_participants = [p for p in participants if p.get("type") == "company"]
        analyst_participants = [p for p in participants if p.get("type") == "analyst"]
        
        with cols[0]:
            st.markdown("#### Company Representatives")
            for participant in company_participants:
                st.markdown(f"- {participant.get('name')} - {participant.get('title', 'N/A')}")
        
        with cols[1]:
            st.markdown("#### Analysts")
            for participant in analyst_participants:
                st.markdown(f"- {participant.get('name')} - {participant.get('company', 'N/A')}")
    
    # Expandable sections for the full transcript
    with st.expander("Full Transcript"):
        sections = transcript.get("sections", [])
        
        if sections:
            for section in sections:
                st.markdown(f"### {section.get('title', 'Section')}")
                st.markdown(section.get("text", "No content available."))
        else:
            text = transcript.get("text", "")
            if text:
                st.markdown(text)
            else:
                st.warning("No transcript content available.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_qa_interface() -> None:
    """Display Q&A interface for asking questions about the earnings call."""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Ask Questions About the Earnings Call</h3>', unsafe_allow_html=True)
    
    query = st.text_input("Enter your question:", placeholder="e.g., What was the CEO's tone on future growth?")
    
    if query and st.button("Ask"):
        # This would integrate with your vector store and LLM components
        # For now, we'll just display a placeholder
        st.info("This feature will integrate with the LLM components to provide answers based on the transcript.")
        st.markdown("Question: " + query)  
        st.markdown("Answer: This functionality requires integration with the vector database and LLM components.")
    
    # Suggested questions
    st.markdown("### Suggested Questions")
    suggested_questions = [
        "What were the key highlights of the earnings call?",
        "How did the CEO discuss future growth prospects?",
        "What were the main challenges mentioned in the call?",
        "How did the company perform compared to analyst expectations?",
        "What guidance did management provide for the next quarter?"
    ]
    
    for question in suggested_questions:
        if st.button(question, key=f"suggested_{slugify(question)}"):
            st.info("This feature will integrate with the LLM components to provide answers based on the transcript.")
            st.markdown("Question: " + question)
            st.markdown("Answer: This functionality requires integration with the vector database and LLM components.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Earnings Call Intelligence Agent</h1>', unsafe_allow_html=True)
    st.markdown("Analyze earnings call transcripts, extract key metrics, assess sentiment, and generate comprehensive reports.")
    
    # Sidebar - Input parameters
    with st.sidebar:
        st.markdown("## 🔍 Search Company")
        search_options = st.radio("Choose search method:", ["By Symbol", "Upcoming Earnings"])
        
        if search_options == "By Symbol":
            symbol = st.text_input("Enter Stock Symbol:", value=st.session_state.current_symbol or "").upper()
            
            current_year = datetime.datetime.now().year
            year = st.selectbox("Select Year:", options=list(range(current_year-5, current_year+1)), index=5)
            quarter = st.selectbox("Select Quarter:", options=[1, 2, 3, 4])
            
            if st.button("Analyze Earnings Call", key="analyze_button"):
                if symbol:
                    success = fetch_data(symbol, quarter, year)
                    if success:
                        st.success(f"Successfully analyzed {symbol} Q{quarter} {year} earnings call!")
                else:
                    st.error("Please enter a stock symbol.")
        
        else:  # Upcoming Earnings
            st.markdown("### Upcoming Earnings Releases")
            
            try:
                with st.spinner("Fetching upcoming earnings..."):
                    upcoming_earnings = get_upcoming_earnings()
                
                if upcoming_earnings:
                    # Convert to DataFrame for better display
                    df = pd.DataFrame(upcoming_earnings)
                    st.dataframe(df[["symbol", "company", "date", "time"]], use_container_width=True)
                    
                    # Allow selection of a company from the list
                    selected_symbol = st.selectbox(
                        "Select a company to analyze previous earnings:",
                        options=[f"{e['symbol']} - {e['company']}" for e in upcoming_earnings]
                    )
                    
                    if selected_symbol:
                        symbol = selected_symbol.split(" - ")[0]
                        
                        current_year = datetime.datetime.now().year
                        year = st.selectbox("Select Year:", options=list(range(current_year-5, current_year+1)), index=5)
                        quarter = st.selectbox("Select Quarter:", options=[1, 2, 3, 4])
                        
                        if st.button("Analyze Previous Earnings Call"):
                            success = fetch_data(symbol, quarter, year)
                            if success:
                                st.success(f"Successfully analyzed {symbol} Q{quarter} {year} earnings call!")
                else:
                    st.warning("No upcoming earnings found.")
            except Exception as e:
                st.error(f"Error fetching upcoming earnings: {str(e)}")
        
        # Export options
        st.markdown("## 📤 Export Options")
        if st.session_state.summary_data:
            export_format = st.selectbox("Select Format:", ["PDF", "Markdown"])
            
            if st.button("Generate Report"):
                symbol = st.session_state.current_symbol
                quarter = st.session_state.current_quarter
                year = st.session_state.current_year
                
                filename = f"{symbol.lower()}_q{quarter}_{year}_report"
                
                if export_format == "PDF":
                    output_path = PDF_OUTPUT_DIR / f"{filename}.pdf"
                    with st.spinner("Generating PDF report..."):
                        success = generate_pdf_report(st.session_state.summary_data, output_path, "reportlab")
                    
                    if success:
                        st.success(f"PDF report generated successfully: {output_path}")
                        # Create a download button if the file exists
                        if os.path.exists(output_path):
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label="Download PDF",
                                    data=file,
                                    file_name=f"{filename}.pdf",
                                    mime="application/pdf"
                                )
                
                elif export_format == "Markdown":
                    markdown_content = generate_markdown_summary(st.session_state.summary_data)
                    st.download_button(
                        label="Download Markdown",
                        data=markdown_content,
                        file_name=f"{filename}.md",
                        mime="text/markdown"
                    )
    
    # Main content area
    if st.session_state.transcript_data and not st.session_state.transcript_data.get("error"):
        # Summary tabs
        tabs = st.tabs(["Overview", "Metrics", "Sentiment", "Comparison", "Transcript", "Q&A"])
        
        with tabs[0]:  # Overview
            st.markdown("## Earnings Call Overview")
            
            # Company header
            company = st.session_state.transcript_data.get("company", st.session_state.current_symbol)
            date = st.session_state.transcript_data.get("date", f"Q{st.session_state.current_quarter} {st.session_state.current_year}")
            
            st.markdown(f"### {company} Earnings Call - {date}")
            
            if st.session_state.summary_data:
                executive_summary = st.session_state.summary_data.get("executive_summary", "")
                st.markdown(f"### Executive Summary")
                st.markdown(executive_summary)
                
                key_points = st.session_state.summary_data.get("key_points", [])
                st.markdown("### Key Takeaways")
                for point in key_points:
                    st.markdown(f"- {point}")
            else:
                st.warning("Summary data not available.")
                
        with tabs[1]:  # Metrics
            st.markdown("## Financial Metrics Analysis")
            if st.session_state.metrics_data:
                display_metrics_card(st.session_state.metrics_data)
            else:
                st.warning("Metrics data not available.")
                
        with tabs[2]:  # Sentiment
            st.markdown("## Sentiment Analysis")
            if st.session_state.sentiment_data:
                display_sentiment_chart(st.session_state.sentiment_data)
            else:
                st.warning("Sentiment data not available.")
                
        with tabs[3]:  # Comparison
            st.markdown("## Quarter-over-Quarter Comparison")
            if st.session_state.comparison_data:
                display_comparison(st.session_state.comparison_data)
            else:
                st.warning("Comparison data not available.")
                
        with tabs[4]:  # Transcript
            st.markdown("## Earnings Call Transcript")
            if st.session_state.transcript_data:
                display_transcript_summary(st.session_state.transcript_data)
            else:
                st.warning("Transcript data not available.")
                
        with tabs[5]:  # Q&A
            st.markdown("## Interactive Q&A")
            display_qa_interface()
    else:
        # Display welcome screen when no data is loaded
        st.markdown("""
        ## Welcome to the Earnings Call Intelligence Agent! 👋
        
        This tool helps you analyze earnings call transcripts to extract valuable insights:
        
        - **Extract Key Metrics**: Revenue, EPS, guidance, and other financial data
        - **Analyze Sentiment**: Gauge management's tone and confidence
        - **Compare Quarters**: See how performance changed over time
        - **Generate Reports**: Create shareable PDF or Markdown summaries
        - **Interactive Q&A**: Ask questions about the earnings call
        
        ### Getting Started
        
        1. Enter a stock symbol in the sidebar
        2. Select the quarter and year
        3. Click "Analyze Earnings Call"
        
        Or browse upcoming earnings releases and select a company to analyze its previous earnings calls.
        """)
        
        # Sample visualization placeholder
        st.markdown("### Sample Visualization")
        sample_data = {
            "Section": ["CEO Comments", "CFO Comments", "Q&A"],
            "Positive": [0.65, 0.45, 0.35],
            "Neutral": [0.25, 0.35, 0.40],
            "Negative": [0.10, 0.20, 0.25]
        }
        
        df = pd.DataFrame(sample_data)
        fig = px.bar(df, x="Section", y=["Positive", "Neutral", "Negative"],
                    title="Sample Sentiment Analysis",
                    labels={"value": "Score", "variable": "Sentiment"},
                    color_discrete_map={
                        "Positive": "#4CAF50",
                        "Neutral": "#FFC107",
                        "Negative": "#F44336"
                    },
                    barmode="group")
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()