# src/analyze/metrics.py
import re
from typing import List, Dict, Any, Union
import json
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_DIR

def extract_financial_metrics(text: str) -> Dict[str, Any]:
    """
    Extract key financial metrics from text using regex patterns.
    
    Args:
        text: Text to extract metrics from
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Revenue patterns
    revenue_patterns = [
        r'revenue\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)',
        r'total\s+revenue\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)',
        r'reported\s+revenue\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)',
    ]
    
    for pattern in revenue_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1))
            # Check for billion/million
            if 'billion' in match.group(0).lower() or 'B' in match.group(0):
                amount *= 1_000_000_000
            elif 'million' in match.group(0).lower() or 'M' in match.group(0):
                amount *= 1_000_000
            metrics['revenue'] = amount
            break
    
    # EPS patterns
    eps_patterns = [
        r'earnings\s+per\s+share\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+\.\d+)',
        r'EPS\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+\.\d+)',
        r'diluted\s+EPS\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+\.\d+)',
    ]
    
    for pattern in eps_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['eps'] = float(match.group(1))
            break
    
    # Net income patterns
    income_patterns = [
        r'net\s+income\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)',
        r'profit\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)',
    ]
    
    for pattern in income_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1))
            # Check for billion/million
            if 'billion' in match.group(0).lower() or 'B' in match.group(0):
                amount *= 1_000_000_000
            elif 'million' in match.group(0).lower() or 'M' in match.group(0):
                amount *= 1_000_000
            metrics['net_income'] = amount
            break
    
    # Gross margin patterns
    margin_patterns = [
        r'gross\s+margin\s+(?:of|was|at|totaled|reached|amounted to)\s+(\d+(?:\.\d+)?)%',
        r'gross\s+margin\s+(?:of|was|at|totaled|reached|amounted to)\s+(\d+(?:\.\d+)?)\s+percent',
    ]
    
    for pattern in margin_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['gross_margin'] = float(match.group(1))
            break
    
    # Operating expenses patterns
    opex_patterns = [
        r'operating\s+expenses\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)',
        r'opex\s+(?:of|was|at|totaled|reached|amounted to)\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)',
    ]
    
    for pattern in opex_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1))
            # Check for billion/million
            if 'billion' in match.group(0).lower() or 'B' in match.group(0):
                amount *= 1_000_000_000
            elif 'million' in match.group(0).lower() or 'M' in match.group(0):
                amount *= 1_000_000
            metrics['operating_expenses'] = amount
            break
    
    # Year-over-year growth patterns
    yoy_patterns = [
        r'(\d+(?:\.\d+)?)%\s+(?:increase|growth|up)\s+(?:year[- ]over[- ]year|compared to last year|YoY)',
        r'(?:increased|grew|up)\s+(\d+(?:\.\d+)?)%\s+(?:year[- ]over[- ]year|compared to last year|YoY)',
    ]
    
    for pattern in yoy_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['yoy_growth'] = float(match.group(1))
            break
    
    return metrics

def extract_guidance(text: str) -> Dict[str, Any]:
    """
    Extract forward-looking guidance from the text.
    
    Args:
        text: Text to extract guidance from
        
    Returns:
        Dictionary of guidance metrics
    """
    guidance = {}
    
    # Look for next quarter revenue guidance
    next_quarter_patterns = [
        r'(?:next quarter|Q\d|upcoming quarter).*?revenue.*?\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)(?:\s+to\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M))?',
        r'revenue(?:\s+guidance)?(?:\s+for)?(?:\s+the)?(?:\s+next|upcoming|following)(?:\s+quarter).*?\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)(?:\s+to\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M))?',
        r'(?:expect|anticipate|project|forecast)(?:\s+next quarter|Q\d)(?:\s+revenue).*?\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)(?:\s+to\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M))?',
    ]
    
    for pattern in next_quarter_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            min_val = float(match.group(1))
            # Check if there's a range
            if match.group(2):
                max_val = float(match.group(2))
                # Convert to appropriate scale
                if 'billion' in match.group(0).lower() or 'B' in match.group(0):
                    min_val *= 1_000_000_000
                    max_val *= 1_000_000_000
                elif 'million' in match.group(0).lower() or 'M' in match.group(0):
                    min_val *= 1_000_000
                    max_val *= 1_000_000
                guidance['next_quarter_revenue'] = [min_val, max_val]
            else:
                # Single value
                if 'billion' in match.group(0).lower() or 'B' in match.group(0):
                    min_val *= 1_000_000_000
                elif 'million' in match.group(0).lower() or 'M' in match.group(0):
                    min_val *= 1_000_000
                guidance['next_quarter_revenue'] = min_val
            break
    
    # Look for full year revenue guidance
    full_year_patterns = [
        r'(?:full year|fiscal year).*?revenue.*?\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)(?:\s+to\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M))?',
        r'revenue(?:\s+guidance)?(?:\s+for)?(?:\s+the)?(?:\s+full|fiscal)(?:\s+year).*?\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)(?:\s+to\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M))?',
        r'(?:expect|anticipate|project|forecast)(?:\s+full year|fiscal year)(?:\s+revenue).*?\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M)(?:\s+to\s+\$?(\d+(?:\.\d+)?)\s+(?:billion|million|B|M))?',
    ]
    
    for pattern in full_year_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            min_val = float(match.group(1))
            # Check if there's a range
            if match.group(2):
                max_val = float(match.group(2))
                # Convert to appropriate scale
                if 'billion' in match.group(0).lower() or 'B' in match.group(0):
                    min_val *= 1_000_000_000
                    max_val *= 1_000_000_000
                elif 'million' in match.group(0).lower() or 'M' in match.group(0):
                    min_val *= 1_000_000
                    max_val *= 1_000_000
                guidance['full_year_revenue'] = [min_val, max_val]
            else:
                # Single value
                if 'billion' in match.group(0).lower() or 'B' in match.group(0):
                    min_val *= 1_000_000_000
                elif 'million' in match.group(0).lower() or 'M' in match.group(0):
                    min_val *= 1_000_000
                guidance['full_year_revenue'] = min_val
            break
    
    # Look for next quarter EPS guidance
    next_quarter_eps_patterns = [
        r'(?:next quarter|Q\d|upcoming quarter).*?EPS.*?\$?(\d+\.\d+)(?:\s+to\s+\$?(\d+\.\d+))?',
        r'EPS(?:\s+guidance)?(?:\s+for)?(?:\s+the)?(?:\s+next|upcoming|following)(?:\s+quarter).*?\$?(\d+\.\d+)(?:\s+to\s+\$?(\d+\.\d+))?',
        r'(?:expect|anticipate|project|forecast)(?:\s+next quarter|Q\d)(?:\s+EPS).*?\$?(\d+\.\d+)(?:\s+to\s+\$?(\d+\.\d+))?',
    ]
    
    for pattern in next_quarter_eps_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            min_val = float(match.group(1))
            # Check if there's a range
            if match.group(2):
                max_val = float(match.group(2))
                guidance['next_quarter_eps'] = [min_val, max_val]
            else:
                # Single value
                guidance['next_quarter_eps'] = min_val
            break
    
    # Look for full year EPS guidance
    full_year_eps_patterns = [
        r'(?:full year|fiscal year).*?EPS.*?\$?(\d+\.\d+)(?:\s+to\s+\$?(\d+\.\d+))?',
        r'EPS(?:\s+guidance)?(?:\s+for)?(?:\s+the)?(?:\s+full|fiscal)(?:\s+year).*?\$?(\d+\.\d+)(?:\s+to\s+\$?(\d+\.\d+))?',
        r'(?:expect|anticipate|project|forecast)(?:\s+full year|fiscal year)(?:\s+EPS).*?\$?(\d+\.\d+)(?:\s+to\s+\$?(\d+\.\d+))?',
    ]
    
    for pattern in full_year_eps_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            min_val = float(match.group(1))
            # Check if there's a range
            if match.group(2):
                max_val = float(match.group(2))
                guidance['full_year_eps'] = [min_val, max_val]
            else:
                # Single value
                guidance['full_year_eps'] = min_val
            break
    
    return guidance

def analyze_transcript_metrics(symbol: str, quarter: int = None, year: int = None) -> Dict[str, Any]:
    """
    Extract metrics from a full transcript.
    
    Args:
        symbol: Stock ticker symbol
        quarter: Fiscal quarter
        year: Fiscal year
        
    Returns:
        Dictionary of all metrics found
    """
    from src.fetch.transcript import get_transcript
    from src.process.chunking import get_sections
    
    # Get the transcript
    transcript = get_transcript(symbol, quarter, year)
    
    if not transcript.get('text'):
        return {"error": "No transcript found"}
    
    # Get transcript text
    text = transcript.get('text', '')
    
    # Get transcript sections
    sections = get_sections(transcript)
    
    # Check if we have cached results
    cache_dir = DATA_DIR / "analysis"
    cache_dir.mkdir(exist_ok=True)
    
    quarter_year = f"_q{quarter}_{year}" if quarter and year else ""
    cache_file = cache_dir / f"{symbol.lower()}{quarter_year}_metrics.json"
    
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # Extract metrics from prepared remarks
    prepared_metrics = extract_financial_metrics(sections.get('prepared_remarks', ''))
    
    # Extract metrics from full text as backup
    full_metrics = extract_financial_metrics(text)
    
    # Combine metrics, preferring prepared remarks
    metrics = {**full_metrics, **prepared_metrics}
    
    # Extract guidance
    guidance = extract_guidance(text)
    
    # Combine results
    results = {
        "symbol": symbol,
        "quarter": quarter,
        "year": year,
        "metrics": metrics,
        "guidance": guidance
    }
    
    # Cache results
    with open(cache_file, 'w') as f:
        json.dump(results, f)
    
    return results

if __name__ == "__main__":
    # Test the metric extraction
    symbol = "AAPL"
    
    print(f"Analyzing metrics for {symbol}...")
    results = analyze_transcript_metrics(symbol)
    
    print("\nExtracted metrics:")
    for key, value in results.get('metrics', {}).items():
        if isinstance(value, (int, float)) and value > 1_000_000:
            # Format large numbers
            if value >= 1_000_000_000:
                print(f"  {key}: ${value / 1_000_000_000:.2f} billion")
            else:
                print(f"  {key}: ${value / 1_000_000:.2f} million")
        else:
            print(f"  {key}: {value}")
    
    print("\nGuidance:")
    for key, value in results.get('guidance', {}).items():
        if isinstance(value, list):
            # Range
            if value[0] > 1_000_000:
                # Format large numbers
                if value[0] >= 1_000_000_000:
                    print(f"  {key}: ${value[0] / 1_000_000_000:.2f} - ${value[1] / 1_000_000_000:.2f} billion")
                else:
                    print(f"  {key}: ${value[0] / 1_000_000:.2f} - ${value[1] / 1_000_000:.2f} million")
            else:
                print(f"  {key}: ${value[0]:.2f} - ${value[1]:.2f}")
        else:
            # Single value
            if isinstance(value, (int, float)) and value > 1_000_000:
                # Format large numbers
                if value >= 1_000_000_000:
                    print(f"  {key}: ${value / 1_000_000_000:.2f} billion")
                else:
                    print(f"  {key}: ${value / 1_000_000:.2f} million")
            else:
                print(f"  {key}: ${value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")