# src/fetch/schedule.py
import requests
import datetime
from typing import List, Dict, Any
import json
import os
from pathlib import Path

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import FINNHUB_API_KEY, DATA_DIR, HEADERS

def get_earnings_calendar(start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
    """
    Get upcoming earnings release dates from Finnhub API.
    
    Args:
        start_date: Start date in format YYYY-MM-DD (default: today)
        end_date: End date in format YYYY-MM-DD (default: 7 days from today)
        
    Returns:
        List of dictionaries with earnings information
    """
    # Set default dates if not provided
    if not start_date:
        start_date = datetime.datetime.now().strftime('%Y-%m-%d')
    if not end_date:
        end_date = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Format dates for API
    from_date = start_date.replace('-', '')
    to_date = end_date.replace('-', '')
    
    # Check if we have an API key
    if not FINNHUB_API_KEY:
        print("Warning: No Finnhub API key provided. Falling back to demo data.")
        return _get_demo_earnings_calendar()
    
    # Make API request
    url = f"https://finnhub.io/api/v1/calendar/earnings"
    params = {
        'from': from_date,
        'to': to_date,
        'token': FINNHUB_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        data = response.json()
        
        # Save the data to a file for caching
        cache_file = DATA_DIR / f"earnings_calendar_{from_date}_to_{to_date}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        return data.get('earningsCalendar', [])
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching earnings calendar: {e}")
        
        # Try to load from cache if available
        cache_file = DATA_DIR / f"earnings_calendar_{from_date}_to_{to_date}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
            return data.get('earningsCalendar', [])
        
        # Fall back to demo data
        return _get_demo_earnings_calendar()

def _get_demo_earnings_calendar() -> List[Dict[str, Any]]:
    """
    Return demo earnings calendar data when API is not available.
    """
    # A few sample earnings dates
    tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    next_week = (datetime.datetime.now() + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    
    return [
        {
            "date": tomorrow,
            "hour": "amc",  # after market close
            "symbol": "AAPL",
            "epsEstimate": 1.43,
            "epsActual": None,
            "revenueEstimate": 93500000000,
            "revenueActual": None,
            "quarter": 1,
            "year": 2025,
        },
        {
            "date": next_week,
            "hour": "bmo",  # before market open
            "symbol": "MSFT",
            "epsEstimate": 2.35,
            "epsActual": None,
            "revenueEstimate": 52300000000,
            "revenueActual": None,
            "quarter": 1,
            "year": 2025,
        },
    ]

def get_upcoming_earnings() -> List[Dict[str, Any]]:
    """
    Get a formatted list of upcoming earnings releases.
    """
    earnings = get_earnings_calendar()
    
    # Format the data for display
    formatted_earnings = []
    for earning in earnings:
        time_of_day = {
            "bmo": "Before Market Open",
            "amc": "After Market Close",
            "dmh": "During Market Hours",
        }.get(earning.get("hour", ""), "Unknown")
        
        formatted_earnings.append({
            "symbol": earning.get("symbol"),
            "company_name": earning.get("name", ""),
            "date": earning.get("date"),
            "time": time_of_day,
            "eps_estimate": earning.get("epsEstimate"),
            "revenue_estimate": earning.get("revenueEstimate"),
            "quarter": f"Q{earning.get('quarter')} {earning.get('year')}",
        })
    
    return formatted_earnings

if __name__ == "__main__":
    # Test the function
    upcoming = get_upcoming_earnings()
    print(f"Found {len(upcoming)} upcoming earnings releases:")
    for i, earning in enumerate(upcoming[:5], 1):  # Print first 5
        print(f"{i}. {earning['symbol']} ({earning['company_name']}): {earning['date']} {earning['time']}")