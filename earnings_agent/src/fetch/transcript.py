# src/fetch/transcript.py
import requests
import yfinance as yf
from bs4 import BeautifulSoup
import re
import time
import os
from pathlib import Path
import json
from typing import Dict, Optional, Any
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import HEADERS, DATA_DIR

# filepath: e:\earnings_agent\src\fetch\transcript.py
# filepath: e:\earnings_agent\src\fetch\transcript.py
def get_transcript(symbol: str, quarter: Optional[int] = None, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get earnings call transcript from multiple sources.
    """
    # Create cache directory if it doesn't exist
    transcript_dir = DATA_DIR / "transcripts"
    transcript_dir.mkdir(exist_ok=True)

    # Check if we have a cached transcript
    cache_file = transcript_dir / f"{symbol.lower()}_q{quarter}_{year}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Try multiple sources in order
    sources = [
        #_get_from_yahoo_finance,  # Removed Yahoo Finance
        _get_from_motley_fool,
        get_real_transcript,
        _get_demo_transcript  # Fallback
    ]

    for source_func in sources:
        try:
            transcript = source_func(symbol, quarter, year)
            if transcript and transcript.get('text'):
                # Cache the transcript
                with open(cache_file, 'w') as f:
                    json.dump(transcript, f)
                return transcript
        except Exception as e:
            print(f"Error with {source_func.__name__}: {e}")
            continue

    print("Falling back to demo transcript...")
    transcript = _get_demo_transcript(symbol, quarter, year) # Pass quarter and year
    print(f"Demo transcript for {transcript['symbol']}")
    print(f"Title: {transcript['title']}")
    print(f"Excerpt: {transcript['text'][:500]}...")
    return transcript

def _get_from_yahoo_finance(symbol: str, quarter: Optional[int] = None, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get transcript from Yahoo Finance earnings calls.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    }
    
    # Get stock info using yfinance
    stock = yf.Ticker(symbol)
    
    # Get earnings call info
    try:
        events = stock.calendar
        if events is not None and 'Earnings Call' in events:
            call_date = events['Earnings Call'].strftime('%Y-%m-%d')
            
            # Try to get the transcript from Yahoo Finance
            url = f"https://finance.yahoo.com/quote/{symbol}/analysis?p={symbol}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                transcript_div = soup.find('div', {'id': 'earnings-call-transcript'})
                
                if transcript_div:
                    return {
                        "symbol": symbol,
                        "source": "Yahoo Finance",
                        "url": url,
                        "date": call_date,
                        "quarter": quarter,
                        "year": year,
                        "title": f"{symbol} Earnings Call Transcript",
                        "text": transcript_div.get_text(separator='\n')
                    }
    
    except Exception as e:
        print(f"Error fetching from Yahoo Finance: {e}")
    
    return {}

def _get_from_motley_fool(symbol: str, quarter: Optional[int] = None, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get transcript from Motley Fool (they sometimes provide free transcripts).
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    }
    
    url = f"https://www.fool.com/earnings/{symbol.lower()}-earnings-call-transcript/"
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            article = soup.find('article', {'class': 'article'})
            
            if article:
                return {
                    "symbol": symbol,
                    "source": "Motley Fool",
                    "url": url,
                    "date": datetime.now().strftime('%Y-%m-%d'),  # Or parse from article
                    "quarter": quarter,
                    "year": year,
                    "title": article.find('h1').get_text() if article.find('h1') else "",
                    "text": article.get_text(separator='\n')
                }
    
    except Exception as e:
        print(f"Error fetching from Motley Fool: {e}")
    
    return {}

def _get_from_alpha_sense(symbol: str, quarter: Optional[int] = None, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get transcript from AlphaSense API (requires API key).
    Note: This is a placeholder - you would need an AlphaSense subscription.
    """
    # You would need to sign up for AlphaSense API access
    # https://www.alpha-sense.com/
    return {}

# filepath: e:\earnings_agent\src\fetch\transcript.py
def _get_demo_transcript(symbol: str, quarter: Optional[int] = None, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Return a demo transcript for testing when real data is unavailable.
    """
    # Path to demo data
    demo_file = DATA_DIR / "demo" / f"{symbol.lower()}_transcript.json"

    # If we have a demo file for this symbol, use it
    if (demo_file.exists()):
        with open(demo_file, 'r') as f:
            return json.load(f)

    # Otherwise return a generic demo transcript
    return {
        "symbol": symbol,
        "source": "Demo",
        "url": "#",
        "date": "April 1, 2025",
        "quarter": quarter or 1,
        "year": year or 2025,
        "title": f"{symbol} Inc (DEMO) Q1 2025 Earnings Call Transcript",
        "text": f"""
        Operator: Good day, and welcome to the {symbol} Inc. First Quarter 2025 Earnings Conference Call. 
        All participants will be in listen-only mode. After today's presentation, there will be an opportunity to ask questions.

        I would now like to turn the conference over to Jane Smith, Head of Investor Relations. Please go ahead.

        Jane Smith: Thank you, operator. Good afternoon, everyone, and welcome to {symbol}'s first quarter 2025 earnings conference call.

        With me on today's call are John Doe, Chief Executive Officer, and Bob Johnson, Chief Financial Officer.

        Before we begin, I'd like to remind you that today's discussion contains forward-looking statements based on the environment as we currently see it, and as such, does include risks and uncertainties.

        Now, I'll turn the call over to John.

        John Doe: Thank you, Jane, and good afternoon, everyone. We're pleased to report a strong start to fiscal year 2025, with first quarter revenue of $10.5 billion, up 15% year-over-year.

        Our performance this quarter demonstrates the continued execution of our strategic initiatives and the robust demand for our products and services.

        Our new product line exceeded expectations, contributing $2.1 billion in revenue, a 25% increase from last quarter.

        We're also seeing strong momentum in our international markets, with revenue growing 18% year-over-year.

        Now, I'll turn it over to Bob to discuss our financial results in more detail.

        Bob Johnson: Thanks, John. As John mentioned, total revenue for the first quarter was $10.5 billion, a 15% increase from the prior year.

        Gross margin was 55%, up 200 basis points from last year, driven by product mix and manufacturing efficiencies.

        Operating expenses were $3.2 billion, representing 30% of revenue, compared to 32% last year.

        Net income was $2.1 billion, resulting in diluted earnings per share of $1.05, up 20% year-over-year.

        Turning to our balance sheet, we ended the quarter with $15 billion in cash and investments, and generated $3 billion in operating cash flow.

        Looking ahead to the second quarter, we expect revenue between $10.8 billion and $11.2 billion, and EPS between $1.10 and $1.15.

        For the full year 2025, we are raising our guidance, with revenue now expected between $44 billion and $45 billion, and EPS between $4.50 and $4.70.

        With that, I'll turn it back to John for closing remarks.

        John Doe: Thanks, Bob. In conclusion, we're very pleased with our start to 2025 and remain confident in our ability to execute on our strategic priorities.

        We're now ready to take your questions. Operator?

        Operator: [Operator Instructions] Our first question comes from Sarah Williams with XYZ Securities.

        Sarah Williams: Thanks for taking my question. Can you provide more color on the performance of your new product line and your expectations for the rest of the year?

        John Doe: Sure, Sarah. As I mentioned, our new product line performed exceptionally well this quarter, with revenue of $2.1 billion. Customer feedback has been very positive, and we're seeing strong adoption rates across all our key markets.

        Based on current trends, we expect this momentum to continue, with the new product line potentially contributing around 25% of our total revenue for the full year.

        Operator: Our next question comes from Michael Brown with ABC Investments.

        Michael Brown: Hi, thanks for taking my question. Could you discuss the impact of recent supply chain challenges on your business, and what steps you're taking to mitigate any issues?

        John Doe: That's a great question, Michael. Like many in our industry, we've experienced some supply chain constraints, particularly for certain components.

        However, we've been proactive in diversifying our supplier base and have built up strategic inventory for critical components.

        As a result, the impact on our business has been relatively minimal, though we continue to monitor the situation closely.

        Bob Johnson: And if I could add to that, Michael, we've factored these considerations into our guidance for the year, so we believe we're well-positioned despite these challenges.

        Operator: This concludes our question-and-answer session. I would like to turn the conference back over to Jane Smith for any closing remarks.

        Jane Smith: Thank you, operator. Thank you all for joining us today. This concludes our earnings call. We look forward to speaking with you again next quarter.
        """
    }

# filepath: e:\earnings_agent\src\fetch\transcript.py
def get_real_transcript(symbol: str, quarter: Optional[int] = None, year: Optional[int] = None) -> Dict[str, Any]:
    """
    Get real earnings call transcript using Financial Modeling Prep API with retry logic.
    """
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        raise ValueError("FMP_API_KEY environment variable not set")

    base_url = "https://financialmodelingprep.com/api/v3"
    url = f"{base_url}/earning_call_transcript/{symbol}?apikey={api_key}"

    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            transcripts = response.json()

            if not transcripts:
                print("No transcripts found")
                return {"error": "No transcripts found"}

            transcript = transcripts[0]  # Most recent

            return {
                "symbol": symbol,
                "source": "Financial Modeling Prep",
                "url": url,
                "date": transcript.get("date"),
                "quarter": transcript.get("quarter"),
                "year": transcript.get("year"),
                "title": transcript.get("title"),
                "text": transcript.get("content"),
                "participants": transcript.get("participants", [])
            }

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error: {e}")
            if e.response.status_code == 403:
                print("Forbidden error. Check your API key and permissions.  This endpoint may require a paid FMP subscription.")
                return {"error": "FMP subscription required for earnings call transcripts."}
            if e.response.status_code == 429:
                print("Rate limit exceeded. Retrying after delay.")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    print("Max retries reached. Falling back to demo transcript.")
    return _get_demo_transcript(symbol, quarter, year)

if __name__ == "__main__":
    # Test the function with a demo symbol
    transcript = get_transcript("AAPL")
    
    if transcript.get('error'):
        print(f"Error: {transcript['error']}")
    else:
        print(f"Got transcript for {transcript['symbol']} from {transcript['source']}")
        print(f"Title: {transcript['title']}")
        print(f"Date: {transcript['date']}")
        print(f"Excerpt: {transcript['text'][:500]}...")
    
    # If there was an error or empty text, fall back to demo
    if not transcript.get('text'):
        print("Falling back to demo transcript...")
        transcript = _get_demo_transcript("AAPL")
        print(f"Demo transcript for {transcript['symbol']}")
        print(f"Title: {transcript['title']}")
        print(f"Excerpt: {transcript['text'][:500]}...")