import sys
import os
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_DIR

from src.analyze.metrics import analyze_transcript_metrics
from src.analyze.sentiment import analyze_transcript_sentiment, generate_sentiment_summary
from src.analyze.comparison import analyze_quarterly_comparison, generate_comparison_summary


def format_number(value: float) -> str:
    """Format a number with appropriate units."""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f} billion"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f} million"
    else:
        return f"${value:.2f}"


def format_percentage(value: float) -> str:
    """Format a percentage value."""
    return f"{value:.2f}%"


def generate_metrics_summary(metrics_data: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of financial metrics.
    
    Args:
        metrics_data: Output from analyze_transcript_metrics
        
    Returns:
        String with formatted summary
    """
    symbol = metrics_data.get("symbol", "").upper()
    quarter = metrics_data.get("quarter")
    year = metrics_data.get("year")
    
    quarter_str = f"Q{quarter} {year}" if quarter and year else ""
    
    metrics = metrics_data.get("metrics", {})
    guidance = metrics_data.get("guidance", {})
    
    summary = [f"# {symbol} {quarter_str} Financial Results\n"]
    
    # Key metrics summary
    metrics_sections = []
    
    if "revenue" in metrics:
        metrics_sections.append("## Revenue")
        revenue = metrics["revenue"]
        metrics_sections.append(f"Revenue was {format_number(revenue)}.")
        
        if "yoy_growth" in metrics:
            growth = metrics["yoy_growth"]
            metrics_sections.append(f" Year-over-year growth was {format_percentage(growth)}.")
    
    if "eps" in metrics:
        metrics_sections.append("\n## Earnings Per Share")
        eps = metrics["eps"]
        metrics_sections.append(f"EPS was ${eps:.2f}.")
    
    if "net_income" in metrics:
        metrics_sections.append("\n## Net Income")
        income = metrics["net_income"]
        metrics_sections.append(f"Net income was {format_number(income)}.")
    
    if "gross_margin" in metrics:
        metrics_sections.append("\n## Gross Margin")
        margin = metrics["gross_margin"]
        metrics_sections.append(f"Gross margin was {margin:.2f}%.")
    
    if "operating_expenses" in metrics:
        metrics_sections.append("\n## Operating Expenses")
        opex = metrics["operating_expenses"]
        metrics_sections.append(f"Operating expenses were {format_number(opex)}.")
    
    if metrics_sections:
        summary.extend(metrics_sections)
    else:
        summary.append("No financial metrics were found in the transcript.")
    
    # Guidance summary
    guidance_sections = []
    
    if guidance:
        guidance_sections.append("\n# Forward Guidance\n")
    
    if "next_quarter_revenue" in guidance:
        guidance_sections.append("## Next Quarter Revenue")
        nq_rev = guidance["next_quarter_revenue"]
        
        if isinstance(nq_rev, list):
            guidance_sections.append(f"Expected to be between {format_number(nq_rev[0])} and {format_number(nq_rev[1])}.")
        else:
            guidance_sections.append(f"Expected to be {format_number(nq_rev)}.")
    
    if "next_quarter_eps" in guidance:
        guidance_sections.append("\n## Next Quarter EPS")
        nq_eps = guidance["next_quarter_eps"]
        
        if isinstance(nq_eps, list):
            guidance_sections.append(f"Expected to be between ${nq_eps[0]:.2f} and ${nq_eps[1]:.2f}.")
        else:
            guidance_sections.append(f"Expected to be ${nq_eps:.2f}.")
    
    if "full_year_revenue" in guidance:
        guidance_sections.append("\n## Full Year Revenue")
        fy_rev = guidance["full_year_revenue"]
        
        if isinstance(fy_rev, list):
            guidance_sections.append(f"Expected to be between {format_number(fy_rev[0])} and {format_number(fy_rev[1])}.")
        else:
            guidance_sections.append(f"Expected to be {format_number(fy_rev)}.")
    
    if "full_year_eps" in guidance:
        guidance_sections.append("\n## Full Year EPS")
        fy_eps = guidance["full_year_eps"]
        
        if isinstance(fy_eps, list):
            guidance_sections.append(f"Expected to be between ${fy_eps[0]:.2f} and ${fy_eps[1]:.2f}.")
        else:
            guidance_sections.append(f"Expected to be ${fy_eps:.2f}.")
    
    if guidance_sections:
        summary.extend(guidance_sections)
    
    return "\n".join(summary)


def generate_executive_quotes(sentiment_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract notable quotes from key executives based on sentiment analysis.
    
    Args:
        sentiment_data: Output from analyze_transcript_sentiment
        
    Returns:
        List of dictionaries with executive quotes
    """
    from src.fetch.transcript import get_transcript
    from src.process.chunking import get_speakers_text
    
    symbol = sentiment_data.get("symbol")
    quarter = sentiment_data.get("quarter")
    year = sentiment_data.get("year")
    
    # Get the transcript
    transcript = get_transcript(symbol, quarter, year)
    
    if not transcript.get('text'):
        return []
    
    # Get text by speaker
    speakers_text = get_speakers_text(transcript)
    
    # Get key executives
    key_executives = sentiment_data.get("key_executives", {})
    
    quotes = []
    
    for speaker in key_executives:
        if speaker in speakers_text:
            text = speakers_text[speaker]
            
            # Split text into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Filter out short sentences
            sentences = [s for s in sentences if len(s) > 30]
            
            # Find most positive sentences (with certain key terms)
            positive_keywords = ["growth", "increase", "success", "strong", "confident", "opportunity"]
            positive_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in positive_keywords):
                    positive_sentences.append(sentence)
            
            # Find most negative sentences (with certain key terms)
            negative_keywords = ["challenge", "decrease", "difficult", "risk", "concern", "pressure"]
            negative_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in negative_keywords):
                    negative_sentences.append(sentence)
            
            # Add quotes to the list
            if positive_sentences:
                quotes.append({
                    "speaker": speaker, 
                    "type": "positive", 
                    "quote": positive_sentences[0]
                })
            
            if negative_sentences:
                quotes.append({
                    "speaker": speaker, 
                    "type": "negative", 
                    "quote": negative_sentences[0]
                })
    
    return quotes


def generate_comprehensive_summary(symbol: str, quarter: int = None, year: int = None) -> Dict[str, Any]:
    """
    Generate a comprehensive earnings call summary including metrics, sentiment, and comparison.
    
    Args:
        symbol: Stock ticker symbol
        quarter: Fiscal quarter
        year: Fiscal year
        
    Returns:
        Dictionary with all summary components
    """
    import re
    
    # Check if we have cached results
    cache_dir = DATA_DIR / "summaries"
    cache_dir.mkdir(exist_ok=True)
    
    quarter_year = f"_q{quarter}_{year}" if quarter and year else ""
    cache_file = cache_dir / f"{symbol.lower()}{quarter_year}_comprehensive_summary.json"
    
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Get metrics analysis
    metrics_data = analyze_transcript_metrics(symbol, quarter, year)
    metrics_summary = generate_metrics_summary(metrics_data)
    
    # Get sentiment analysis
    sentiment_data = analyze_transcript_sentiment(symbol, quarter, year)
    sentiment_summary = generate_sentiment_summary(sentiment_data)
    
    # Try to get comparison data if quarter and year are provided
    comparison_summary = ""
    if quarter is not None and year is not None:
        try:
            comparison_data = analyze_quarterly_comparison(symbol, quarter, year)
            comparison_summary = generate_comparison_summary(comparison_data)
        except Exception as e:
            comparison_summary = f"# Quarter-over-Quarter Comparison\n\nUnable to generate comparison: {str(e)}"
    
    # Get executive quotes
    quotes = generate_executive_quotes(sentiment_data)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Combine results
    results = {
        "symbol": symbol,
        "quarter": quarter,
        "year": year,
        "metrics_summary": metrics_summary,
        "sentiment_summary": sentiment_summary,
        "comparison_summary": comparison_summary,
        "executive_quotes": quotes,
        "generated_at": timestamp
    }
    
    # Cache results
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    
    return results


def generate_markdown_summary(summary_data: Dict[str, Any]) -> str:
    """
    Generate a full markdown report from comprehensive summary data.
    
    Args:
        summary_data: Output from generate_comprehensive_summary
        
    Returns:
        Markdown formatted string
    """
    symbol = summary_data.get("symbol", "").upper()
    quarter = summary_data.get("quarter")
    year = summary_data.get("year")
    timestamp = summary_data.get("generated_at")
    
    quarter_str = f"Q{quarter} {year}" if quarter and year else ""
    
    sections = []
    
    # Title
    sections.append(f"# {symbol} {quarter_str} Earnings Call Summary\n")
    sections.append(f"*Generated on {timestamp}*\n")
    
    # Table of contents
    sections.append("## Table of Contents\n")
    sections.append("1. [Financial Results](#financial-results)")
    sections.append("2. [Sentiment Analysis](#sentiment-analysis)")
    if summary_data.get("comparison_summary"):
        sections.append("3. [Quarter-over-Quarter Comparison](#quarter-over-quarter-comparison)")
    sections.append("3. [Key Executive Quotes](#key-executive-quotes)\n")
    
    # Financial results
    sections.append("## Financial Results\n")
    metrics_summary = summary_data.get("metrics_summary", "").split("\n", 1)
    if len(metrics_summary) > 1:
        sections.append(metrics_summary[1])  # Skip the title as we already have one
    else:
        sections.append("*No financial data available.*")
    
    # Sentiment analysis
    sections.append("\n## Sentiment Analysis\n")
    sentiment_summary = summary_data.get("sentiment_summary", "").split("\n", 1)
    if len(sentiment_summary) > 1:
        sections.append(sentiment_summary[1])  # Skip the title
    else:
        sections.append("*No sentiment analysis available.*")
    
    # Comparison
    if summary_data.get("comparison_summary"):
        sections.append("\n## Quarter-over-Quarter Comparison\n")
        comparison_summary = summary_data.get("comparison_summary", "").split("\n", 1)
        if len(comparison_summary) > 1:
            sections.append(comparison_summary[1])  # Skip the title
    
    # Executive quotes
    sections.append("\n## Key Executive Quotes\n")
    quotes = summary_data.get("executive_quotes", [])
    
    if quotes:
        for quote in quotes:
            speaker = quote.get("speaker", "")
            quote_type = quote.get("type", "")
            quote_text = quote.get("quote", "")
            
            sections.append(f"### {speaker}\n")
            if quote_type == "positive":
                sections.append("*Positive statement:*\n")
            else:
                sections.append("*Concerning statement:*\n")
            
            sections.append(f"> {quote_text}\n")
    else:
        sections.append("*No notable executive quotes found.*")
    
    return "\n".join(sections)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate earnings call summary')
    parser.add_argument('symbol', type=str, help='Stock ticker symbol')
    parser.add_argument('--quarter', type=int, help='Fiscal quarter (1-4)')
    parser.add_argument('--year', type=int, help='Fiscal year')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    print(f"Generating comprehensive summary for {args.symbol}...")
    summary_data = generate_comprehensive_summary(args.symbol, args.quarter, args.year)
    
    markdown_report = generate_markdown_summary(summary_data)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(markdown_report)
        print(f"Summary saved to {args.output}")
    else:
        print(markdown_report)