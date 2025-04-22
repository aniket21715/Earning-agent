# src/process/chunking.py
from typing import List, Dict, Any
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import CHUNK_SIZE, CHUNK_OVERLAP

def clean_transcript(text: str) -> str:
    """
    Clean up the transcript text.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned transcript text
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR/scraping errors
    text = re.sub(r'Q:', '\nQ:', text)  # Ensure "Q:" starts on a new line
    text = re.sub(r'A:', '\nA:', text)  # Ensure "A:" starts on a new line
    
    return text.strip()

def get_speaker_from_line(line: str) -> str:
    """
    Extract speaker name from a line of text.
    
    Args:
        line: Line of text
        
    Returns:
        Speaker name or empty string
    """
    # Look for patterns like "John Doe: " or "John Doe, CEO: "
    speaker_match = re.match(r'^([^:]+):', line)
    if speaker_match:
        return speaker_match.group(1).strip()
    return ""

def split_transcript_into_chunks(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a transcript into manageable chunks for processing.
    
    Args:
        transcript: Dictionary with transcript data
        
    Returns:
        List of chunk dictionaries
    """
    text = transcript.get('text', '')
    if not text:
        return []
    
    # Clean the text
    clean_text = clean_transcript(text)
    
    # Create the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split the text
    chunks = text_splitter.split_text(clean_text)
    
    # Create chunk dictionaries with metadata
    chunk_dicts = []
    for i, chunk in enumerate(chunks):
        # Try to identify the speaker(s) in this chunk
        speakers = set()
        for line in chunk.split('\n'):
            speaker = get_speaker_from_line(line)
            if speaker:
                speakers.add(speaker)
        
        chunk_dicts.append({
            "id": i,
            "text": chunk,
            "symbol": transcript.get('symbol', ''),
            "speakers": list(speakers),
            "date": transcript.get('date', ''),
            "quarter": transcript.get('quarter', ''),
            "year": transcript.get('year', ''),
            "source": transcript.get('source', '')
        })
    
    return chunk_dicts

def get_sections(transcript: Dict[str, Any]) -> Dict[str, str]:
    """
    Attempt to identify key sections of the transcript.
    
    Args:
        transcript: Dictionary with transcript data
        
    Returns:
        Dictionary with sections
    """
    text = transcript.get('text', '')
    if not text:
        return {}
    
    # Clean the text
    clean_text = clean_transcript(text)
    
    # Try to identify the prepared remarks section
    prepared_remarks_match = re.search(r'(?:opening remarks|prepared remarks|Management Discussion).*?(?=Q&A|Question-and-Answer|Question and Answer|\Z)', 
                                       clean_text, re.DOTALL | re.IGNORECASE)
    prepared_remarks = prepared_remarks_match.group(0) if prepared_remarks_match else ""
    
    # Try to identify the Q&A section
    qa_match = re.search(r'(?:Q&A|Question-and-Answer|Question and Answer).*', clean_text, re.DOTALL | re.IGNORECASE)
    qa_section = qa_match.group(0) if qa_match else ""
    
    # Try to identify the operator opening
    opening_match = re.search(r'^.*?(?=prepared remarks|opening remarks|\n\n)', clean_text, re.DOTALL | re.IGNORECASE)
    opening = opening_match.group(0) if opening_match else ""
    
    return {
        "opening": opening.strip(),
        "prepared_remarks": prepared_remarks.strip(),
        "qa": qa_section.strip(),
    }

def get_speakers_text(transcript: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract text by speaker from the transcript.
    
    Args:
        transcript: Dictionary with transcript data
        
    Returns:
        Dictionary mapping speaker names to their text
    """
    text = transcript.get('text', '')
    if not text:
        return {}
    
    # Clean the text
    clean_text = clean_transcript(text)
    
    # Split into lines
    lines = clean_text.split('\n')
    
    # Process line by line to group text by speaker
    speakers_text = {}
    current_speaker = None
    current_text = []
    
    for line in lines:
        # Try to identify speaker at start of line
        speaker = get_speaker_from_line(line)
        
        if speaker:
            # If we found a new speaker, save previous speaker's text
            if current_speaker and current_text:
                speakers_text[current_speaker] = ' '.join(current_text)
            
            # Start collecting text for new speaker
            current_speaker = speaker
            current_text = [line.replace(f"{speaker}:", "").strip()]
        elif current_speaker and line.strip():
            # Continue collecting text for current speaker
            current_text.append(line.strip())
    
    # Don't forget to save the last speaker's text
    if current_speaker and current_text:
        speakers_text[current_speaker] = ' '.join(current_text)
    
    return speakers_text

if __name__ == "__main__":
    # Test with demo data
    from src.fetch.transcript import _get_demo_transcript
    
    transcript = _get_demo_transcript("AAPL")
    
    # Test chunking
    chunks = split_transcript_into_chunks(transcript)
    print(f"Split transcript into {len(chunks)} chunks")
    print(f"First chunk: {chunks[0]['text'][:200]}...")
    print(f"Speakers in first chunk: {chunks[0]['speakers']}")
    
    # Test section extraction
    sections = get_sections(transcript)
    for section, content in sections.items():
        print(f"\n--- {section.upper()} ---")
        print(f"{content[:200]}...")