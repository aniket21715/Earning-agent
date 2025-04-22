import re
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_DIR

# Dictionary of positive financial terms
POSITIVE_TERMS = [
    "strong", "growth", "increase", "improved", "exceeded", "record", "success", 
    "opportunity", "positive", "gain", "higher", "beat", "above", "confident",
    "robust", "impressive", "surpass", "outperform", "upside", "favorable", 
    "optimistic", "encouraged", "pleased", "excited", "happy", "bullish",
    "momentum", "accelerate", "advantage", "efficient", "achievement"
]

# Dictionary of negative financial terms
NEGATIVE_TERMS = [
    "decline", "decrease", "lower", "weak", "miss", "below", "challenge", "concern",
    "difficult", "down", "slowdown", "underperform", "disappointing", "cautious",
    "risk", "uncertainty", "pressure", "headwind", "struggle", "unexpected",
    "deteriorate", "loss", "adverse", "bearish", "worrisome", "downside",
    "worried", "obstacle", "problem", "issue", "difficult", "negative"
]

# Contextual negations that can flip sentiment
NEGATIONS = [
    "not", "no", "never", "neither", "nor", "none", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "didn't",
    "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't"
]

# Phrases that have specific sentiment
POSITIVE_PHRASES = [
    "ahead of expectations", "exceeded expectations", "better than expected",
    "record quarter", "strong performance", "robust growth", "favorable results",
    "positive outlook", "strong demand", "strategic advantage", "market share gains"
]

NEGATIVE_PHRASES = [
    "below expectations", "missed expectations", "worse than expected",
    "challenging quarter", "disappointing results", "downward trend",
    "negative outlook", "weakening demand", "competitive pressures", "market share losses"
]

def simple_sentiment_analysis(text: str) -> Dict[str, Any]:
    """
    Perform simple rule-based sentiment analysis on text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    # Normalize text
    text = text.lower()
    
    # Count positive and negative terms
    positive_count = sum(1 for term in POSITIVE_TERMS if re.search(r'\b' + term + r'\b', text))
    negative_count = sum(1 for term in NEGATIVE_TERMS if re.search(r'\b' + term + r'\b', text))
    
    # Check for positive phrases
    for phrase in POSITIVE_PHRASES:
        phrase_count = len(re.findall(phrase, text, re.IGNORECASE))
        positive_count += phrase_count * 2  # Weight phrases more heavily
    
    # Check for negative phrases
    for phrase in NEGATIVE_PHRASES:
        phrase_count = len(re.findall(phrase, text, re.IGNORECASE))
        negative_count += phrase_count * 2  # Weight phrases more heavily
    
    # Calculate total term count
    total_count = positive_count + negative_count
    
    # Calculate sentiment score (range from -1 to 1)
    if total_count > 0:
        sentiment_score = (positive_count - negative_count) / total_count
    else:
        sentiment_score = 0.0
    
    # Get most common positive and negative terms
    positive_matches = []
    for term in POSITIVE_TERMS:
        matches = re.findall(r'\b' + term + r'\b', text)
        if matches:
            positive_matches.append((term, len(matches)))
    
    negative_matches = []
    for term in NEGATIVE_TERMS:
        matches = re.findall(r'\b' + term + r'\b', text)
        if matches:
            negative_matches.append((term, len(matches)))
    
    # Sort by frequency
    positive_matches.sort(key=lambda x: x[1], reverse=True)
    negative_matches.sort(key=lambda x: x[1], reverse=True)

    return {
        "sentiment_score": sentiment_score,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_matches": positive_matches[:5],  # Top 5 most frequent
        "negative_matches": negative_matches[:5]   # Top 5 most frequent
    }


def analyze_speaker_sentiment(speakers_text: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze sentiment for each speaker in the transcript.
    
    Args:
        speakers_text: Dictionary mapping speaker names to their text
        
    Returns:
        Dictionary mapping speaker names to sentiment analysis
    """
    results = {}
    
    for speaker, text in speakers_text.items():
        results[speaker] = simple_sentiment_analysis(text)
    
    return results


def analyze_transcript_sentiment(symbol: str, quarter: int = None, year: int = None) -> Dict[str, Any]:
    """
    Analyze sentiment for a full transcript, including by speaker and by section.
    
    Args:
        symbol: Stock ticker symbol
        quarter: Fiscal quarter
        year: Fiscal year
        
    Returns:
        Dictionary with sentiment analysis results
    """
    from src.fetch.transcript import get_transcript
    from src.process.chunking import get_sections, get_speakers_text
    
    # Get the transcript
    transcript = get_transcript(symbol, quarter, year)
    
    if not transcript.get('text'):
        return {"error": "No transcript found"}
    
    # Check if we have cached results
    cache_dir = DATA_DIR / "analysis"
    cache_dir.mkdir(exist_ok=True)
    
    quarter_year = f"_q{quarter}_{year}" if quarter and year else ""
    cache_file = cache_dir / f"{symbol.lower()}{quarter_year}_sentiment.json"
    
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Get transcript text
    text = transcript.get('text', '')
    
    # Get transcript sections
    sections = get_sections(transcript)
    
    # Get text by speaker
    speakers_text = get_speakers_text(transcript)
    
    # Analyze overall sentiment
    overall_sentiment = simple_sentiment_analysis(text)
    
    # Analyze sentiment by section
    section_sentiment = {}
    for section_name, section_text in sections.items():
        if section_text:
            section_sentiment[section_name] = simple_sentiment_analysis(section_text)
    
    # Analyze sentiment by speaker
    speaker_sentiment = analyze_speaker_sentiment(speakers_text)
    
    # Identify key executives
    key_executives = {}
    for speaker in speakers_text:
        # Look for CEO, CFO, etc. in the speaker name
        if any(role in speaker for role in ["CEO", "Chief Executive", "CFO", "Chief Financial", "COO", "Chief Operating"]):
            key_executives[speaker] = speaker_sentiment.get(speaker, {})
    
    # Analyze sentiment trend over time by splitting the transcript into quarters
    trend_analysis = {}
    if text:
        # Split text into roughly equal parts to analyze trend
        parts = 4  # Four quarters to analyze trend
        text_length = len(text)
        chunk_size = text_length // parts
        
        for i in range(parts):
            start = i * chunk_size
            end = start + chunk_size if i < parts - 1 else text_length
            chunk_text = text[start:end]
            trend_analysis[f"part_{i+1}"] = simple_sentiment_analysis(chunk_text)
    
    # Combine results
    results = {
        "symbol": symbol,
        "quarter": quarter,
        "year": year,
        "overall_sentiment": overall_sentiment,
        "section_sentiment": section_sentiment,
        "speaker_sentiment": speaker_sentiment,
        "key_executives": key_executives,
        "sentiment_trend": trend_analysis
    }
    
    # Cache results
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    
    return results


def generate_sentiment_summary(sentiment_data: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of sentiment analysis.
    
    Args:
        sentiment_data: Output from analyze_transcript_sentiment
        
    Returns:
        String with formatted summary
    """
    symbol = sentiment_data.get("symbol", "").upper()
    quarter = sentiment_data.get("quarter")
    year = sentiment_data.get("year")
    
    quarter_str = f"Q{quarter} {year}" if quarter and year else ""
    
    summary = [f"# {symbol} {quarter_str} Earnings Call Sentiment Analysis\n"]
    
    # Overall sentiment
    overall = sentiment_data.get("overall_sentiment", {})
    score = overall.get("sentiment_score", 0)
    
    sentiment_label = "neutral"
    if score > 0.2:
        sentiment_label = "positive"
    elif score > 0.5:
        sentiment_label = "strongly positive"
    elif score < -0.2:
        sentiment_label = "negative"
    elif score < -0.5:
        sentiment_label = "strongly negative"
    
    summary.append(f"## Overall Sentiment: {sentiment_label.title()}\n")
    summary.append(f"The overall tone of the earnings call was {sentiment_label} with a sentiment score of {score:.2f}.\n")
    
    # Top positive and negative terms
    positive_matches = overall.get("positive_matches", [])
    negative_matches = overall.get("negative_matches", [])
    
    if positive_matches:
        summary.append("### Top Positive Terms:\n")
        for term, count in positive_matches[:3]:
            summary.append(f"- \"{term}\" (mentioned {count} times)\n")
    
    if negative_matches:
        summary.append("### Top Negative Terms:\n")
        for term, count in negative_matches[:3]:
            summary.append(f"- \"{term}\" (mentioned {count} times)\n")
    
    # Key executives sentiment
    key_execs = sentiment_data.get("key_executives", {})
    if key_execs:
        summary.append("\n## Key Executives Sentiment\n")
        for exec_name, exec_sentiment in key_execs.items():
            exec_score = exec_sentiment.get("sentiment_score", 0)
            
            exec_sentiment_label = "neutral"
            if exec_score > 0.2:
                exec_sentiment_label = "positive"
            elif exec_score > 0.5:
                exec_sentiment_label = "strongly positive"
            elif exec_score < -0.2:
                exec_sentiment_label = "negative"
            elif exec_score < -0.5:
                exec_sentiment_label = "strongly negative"
                
            summary.append(f"### {exec_name}: {exec_sentiment_label.title()} ({exec_score:.2f})\n")
            
            # Top terms for this executive
            pos_terms = exec_sentiment.get("positive_matches", [])
            neg_terms = exec_sentiment.get("negative_matches", [])
            
            if pos_terms:
                summary.append(f"Frequently used positive terms: {', '.join([term for term, _ in pos_terms[:3]])}\n")
            
            if neg_terms:
                summary.append(f"Frequently used negative terms: {', '.join([term for term, _ in neg_terms[:3]])}\n")
    
    # Section sentiment
    section_sentiment = sentiment_data.get("section_sentiment", {})
    if "prepared_remarks" in section_sentiment and "qa_session" in section_sentiment:
        prepared_score = section_sentiment["prepared_remarks"].get("sentiment_score", 0)
        qa_score = section_sentiment["qa_session"].get("sentiment_score", 0)
        
        summary.append("\n## Section Sentiment\n")
        summary.append(f"Prepared remarks sentiment score: {prepared_score:.2f}\n")
        summary.append(f"Q&A session sentiment score: {qa_score:.2f}\n")
        
        if prepared_score > qa_score + 0.1:
            summary.append("\nThe prepared remarks were notably more positive than the Q&A session, which may indicate management was more guarded when responding to analyst questions.\n")
        elif qa_score > prepared_score + 0.1:
            summary.append("\nThe Q&A session was more positive than the prepared remarks, suggesting management provided reassuring responses to analyst concerns.\n")
    
    # Sentiment trend
    trend = sentiment_data.get("sentiment_trend", {})
    if trend and len(trend) >= 2:
        summary.append("\n## Sentiment Trend During Call\n")
        
        # Get scores in order
        ordered_scores = []
        for i in range(1, len(trend) + 1):
            key = f"part_{i}"
            if key in trend:
                ordered_scores.append(trend[key].get("sentiment_score", 0))
        
        if ordered_scores:
            start_score = ordered_scores[0]
            end_score = ordered_scores[-1]
            
            if end_score > start_score + 0.1:
                summary.append("The sentiment improved as the call progressed, suggesting a positive turn in the discussion.\n")
            elif start_score > end_score + 0.1:
                summary.append("The sentiment declined as the call progressed, which may indicate challenges raised during the discussion.\n")
            else:
                summary.append("The sentiment remained relatively consistent throughout the call.\n")
    
    return "\n".join(summary)


if __name__ == "__main__":
    # Test the sentiment analysis
    symbol = "AAPL"
    
    print(f"Analyzing sentiment for {symbol}...")
    sentiment = analyze_transcript_sentiment(symbol)
    
    summary = generate_sentiment_summary(sentiment)
    print(summary)
    
    # Print detailed sentiment data
    print("\nOverall sentiment score:", sentiment.get("overall_sentiment", {}).get("sentiment_score", 0))
    
    print("\nSentiment by speaker:")
    print("\nSentiment by speaker:")
    for speaker, data in sentiment.get("speaker_sentiment", {}).items():
        score = data.get("sentiment_score", 0)
        print(f"  {speaker}: {score:.2f}")
    
    print("\nSentiment trend during call:")
    for part, data in sentiment.get("sentiment_trend", {}).items():
        score = data.get("sentiment_score", 0)
        print(f"  {part}: {score:.2f}")
    
    print("\nSection sentiment:")
    for section, data in sentiment.get("section_sentiment", {}).items():
        score = data.get("sentiment_score", 0)
        print(f"  {section}: {score:.2f}")