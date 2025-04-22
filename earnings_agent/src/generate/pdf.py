import sys
import os
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from datetime import datetime
import io
import re
import base64
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import DATA_DIR

from src.generate.summary import generate_comprehensive_summary, generate_markdown_summary

try:
    import reportlab
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, ListFlowable, ListItem
    from reportlab.platypus.flowables import HRFlowable
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
except ImportError:
    print("ReportLab not installed. Install it with: pip install reportlab")
    reportlab = None

try:
    import markdown
    from xhtml2pdf import pisa
except ImportError:
    print("markdown and/or xhtml2pdf not installed. Install with: pip install markdown xhtml2pdf")
    markdown = None
    pisa = None


def markdown_to_pdf(md_text: str, output_path: str) -> bool:
    """
    Convert markdown text to PDF using xhtml2pdf.
    
    Args:
        md_text: Markdown formatted text
        output_path: Path to save the PDF
        
    Returns:
        True if successful, False otherwise
    """
    if markdown is None or pisa is None:
        print("markdown and/or xhtml2pdf not installed. Cannot convert markdown to PDF.")
        return False
    
    # Convert markdown to HTML
    html = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code']
    )
    
    # Add some CSS for better formatting
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; }}
            h1 {{ color: #333366; }}
            h2 {{ color: #333366; border-bottom: 1px solid #cccccc; padding-bottom: 5px; }}
            h3 {{ color: #333366; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
            th {{ background-color: #f2f2f2; }}
            blockquote {{ background-color: #f9f9f9; border-left: 5px solid #cccccc; padding: 10px; margin: 10px 0; }}
            code {{ background-color: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }}
        </style>
    </head>
    <body>
    {html}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    with open(output_path, "wb") as output_file:
        result = pisa.CreatePDF(io.StringIO(html), dest=output_file)
    
    return result.err == 0


def create_sentiment_chart(sentiment_data: Dict[str, Any]) -> Optional[str]:
    """
    Create a chart visualizing sentiment trends during the call.
    
    Args:
        sentiment_data: Sentiment analysis data
        
    Returns:
        Path to the saved chart image or None if failed
    """
    try:
        trend_data = sentiment_data.get("sentiment_trend", {})
        if not trend_data or len(trend_data) < 2:
            return None
            
        # Extract data
        labels = []
        scores = []
        
        for i in range(1, len(trend_data) + 1):
            key = f"part_{i}"
            if key in trend_data:
                labels.append(f"Part {i}")
                scores.append(trend_data[key].get("sentiment_score", 0))
        
        if not scores:
            return None
            
        # Create chart
        plt.figure(figsize=(10, 5))
        plt.plot(labels, scores, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.title('Sentiment Trend During Earnings Call', fontsize=16)
        plt.ylabel('Sentiment Score (-1 to +1)', fontsize=12)
        plt.ylim([-1, 1])
        
        # Add color to background based on sentiment
        for i in range(len(scores)-1):
            if scores[i] > 0:
                plt.axvspan(i, i+1, alpha=0.1, color='green')
            elif scores[i] < 0:
                plt.axvspan(i, i+1, alpha=0.1, color='red')
        
        # Save chart
        chart_dir = DATA_DIR / "charts"
        chart_dir.mkdir(exist_ok=True)
        
        symbol = sentiment_data.get("symbol", "").lower()
        quarter = sentiment_data.get("quarter")
        year = sentiment_data.get("year")
        
        quarter_year = f"_q{quarter}_{year}" if quarter and year else ""
        chart_path = chart_dir / f"{symbol}{quarter_year}_sentiment_trend.png"
        
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    except Exception as e:
        print(f"Error creating sentiment chart: {e}")
        return None


def create_speaker_sentiment_chart(sentiment_data: Dict[str, Any]) -> Optional[str]:
    """
    Create a chart comparing sentiment across different speakers.
    
    Args:
        sentiment_data: Sentiment analysis data
        
    Returns:
        Path to the saved chart image or None if failed
    """
    try:
        speaker_data = sentiment_data.get("speaker_sentiment", {})
        if not speaker_data or len(speaker_data) < 2:
            return None
            
        # Filter to include only key speakers with enough content
        key_speakers = {}
        for speaker, data in speaker_data.items():
            total_terms = data.get("positive_count", 0) + data.get("negative_count", 0)
            if total_terms > 5:  # Only include speakers with sufficient content
                # Shorten very long speaker names
                short_name = speaker
                if len(speaker) > 20:
                    parts = speaker.split()
                    if len(parts) > 1:
                        short_name = f"{parts[0]} {parts[-1]}"
                
                key_speakers[short_name] = data.get("sentiment_score", 0)
        
        if len(key_speakers) < 2:
            return None
            
        # Sort speakers by sentiment score
        sorted_speakers = sorted(key_speakers.items(), key=lambda x: x[1])
        speakers = [item[0] for item in sorted_speakers]
        scores = [item[1] for item in sorted_speakers]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, max(4, len(speakers) * 0.5)))
        
        # Create colormap based on sentiment
        colors = ['#FF6B6B' if score < 0 else '#4ECDC4' for score in scores]
        
        plt.barh(speakers, scores, color=colors)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3, axis='x')
        plt.title('Sentiment by Speaker', fontsize=16)
        plt.xlabel('Sentiment Score (-1 to +1)', fontsize=12)
        plt.xlim([-1, 1])
        
        # Save chart
        chart_dir = DATA_DIR / "charts"
        chart_dir.mkdir(exist_ok=True)
        
        symbol = sentiment_data.get("symbol", "").lower()
        quarter = sentiment_data.get("quarter")
        year = sentiment_data.get("year")
        
        quarter_year = f"_q{quarter}_{year}" if quarter and year else ""
        chart_path = chart_dir / f"{symbol}{quarter_year}_speaker_sentiment.png"
        
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    except Exception as e:
        print(f"Error creating speaker sentiment chart: {e}")
        return None


def create_metrics_comparison_chart(comparison_data: Dict[str, Any]) -> Optional[str]:
    """
    Create a chart comparing key metrics between quarters.
    
    Args:
        comparison_data: Comparison analysis data
        
    Returns:
        Path to the saved chart image or None if failed
    """
    try:
        comp = comparison_data.get("comparison", {})
        if not comp:
            return None
        
        # Get quarterly data for key metrics
        metrics = []
        current_values = []
        previous_values = []
        pct_changes = []
        
        for metric in ["revenue", "net_income", "eps", "gross_margin"]:
            if f"{metric}_pct_change" in comp:
                metrics.append(metric.replace("_", " ").title())
                
                # Add current and previous values
                current_key = f"current_{metric}"
                previous_key = f"previous_{metric}"
                
                # Normalize values for display
                if metric in ["revenue", "net_income"]:
                    # Convert to billions for display
                    current_values.append(comp.get(current_key, 0) / 1_000_000_000)
                    previous_values.append(comp.get(previous_key, 0) / 1_000_000_000)
                else:
                    current_values.append(comp.get(current_key, 0))
                    previous_values.append(comp.get(previous_key, 0))
                
                pct_changes.append(comp.get(f"{metric}_pct_change", 0))
        
        if not metrics:
            return None
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Bar chart for values
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, current_values, width, label='Current Quarter')
        ax1.bar(x + width/2, previous_values, width, label='Previous Quarter')
        
        ax1.set_ylabel('Value')
        ax1.set_title('Quarterly Financial Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate(current_values):
            if metrics[i] in ["Revenue", "Net Income"]:
                ax1.text(i - width/2, v + 0.05, f"${v:.1f}B", ha='center')
            elif metrics[i] == "Eps":
                ax1.text(i - width/2, v + 0.05, f"${v:.2f}", ha='center')
            else:
                ax1.text(i - width/2, v + 0.05, f"{v:.1f}%", ha='center')
                
        for i, v in enumerate(previous_values):
            if metrics[i] in ["Revenue", "Net Income"]:
                ax1.text(i + width/2, v + 0.05, f"${v:.1f}B", ha='center')
            elif metrics[i] == "Eps":
                ax1.text(i + width/2, v + 0.05, f"${v:.2f}", ha='center')
            else:
                ax1.text(i + width/2, v + 0.05, f"{v:.1f}%", ha='center')
        
        # Bar chart for percentage changes
        colors = ['#4ECDC4' if x >= 0 else '#FF6B6B' for x in pct_changes]
        ax2.bar(x, pct_changes, color=colors)
        ax2.set_ylabel('% Change')
        ax2.set_title('Quarter-over-Quarter Percentage Changes')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add percentage labels on top of bars
        for i, v in enumerate(pct_changes):
            if v >= 0:
                ax2.text(i, v + 0.5, f"+{v:.1f}%", ha='center')
            else:
                ax2.text(i, v - 2.0, f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        
        # Save chart
        chart_dir = DATA_DIR / "charts"
        chart_dir.mkdir(exist_ok=True)
        
        symbol = comparison_data.get("symbol", "").lower()
        current_quarter = comparison_data.get("current_quarter")
        current_year = comparison_data.get("current_year")
        previous_quarter = comparison_data.get("previous_quarter")
        previous_year = comparison_data.get("previous_year")
        
        chart_path = chart_dir / f"{symbol}_q{current_quarter}_{current_year}_vs_q{previous_quarter}_{previous_year}_metrics.png"
        
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    except Exception as e:
        print(f"Error creating metrics comparison chart: {e}")
        return None


def generate_pdf_report_with_reportlab(summary_data: Dict[str, Any], output_path: str) -> bool:
    """
    Create a PDF report using ReportLab with charts and formatted text.
    
    Args:
        summary_data: Comprehensive summary data
        output_path: Path to save the PDF
        
    Returns:
        True if successful, False otherwise
    """
    if reportlab is None:
        print("ReportLab not installed. Cannot generate PDF report.")
        return False
    
    try:
        symbol = summary_data.get("symbol", "").upper()
        quarter = summary_data.get("quarter")
        year = summary_data.get("year")
        timestamp = summary_data.get("generated_at")
        
        quarter_str = f"Q{quarter} {year}" if quarter and year else ""
        
        # Get sentiment data to create charts
        sentiment_data = None
        comparison_data = None
        
        if quarter and year:
            from src.analyze.sentiment import analyze_transcript_sentiment
            from src.analyze.comparison import analyze_quarterly_comparison
            
            sentiment_data = analyze_transcript_sentiment(symbol, quarter, year)
            try:
                comparison_data = analyze_quarterly_comparison(symbol, quarter, year)
            except Exception:
                comparison_data = None
        
        # Create charts
        sentiment_chart = None
        speaker_chart = None
        metrics_chart = None
        
        if sentiment_data:
            sentiment_chart = create_sentiment_chart(sentiment_data)
            speaker_chart = create_speaker_sentiment_chart(sentiment_data)
        
        if comparison_data:
            metrics_chart = create_metrics_comparison_chart(comparison_data)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=inch * 0.5,
            leftMargin=inch * 0.5,
            topMargin=inch * 0.5,
            bottomMargin=inch * 0.5
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='Title',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#333366')
        ))
        
        styles.add(ParagraphStyle(
            name='Heading2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.HexColor('#333366')
        ))
        
        styles.add(ParagraphStyle(
            name='Heading3',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            textColor=colors.HexColor('#333366')
        ))
        
        styles.add(ParagraphStyle(
            name='Normal',
            parent=styles['Normal'],    
            fontSize=10,
            leading=12,
            spaceAfter=6,
            textColor=colors.black
        ))
        styles.add(ParagraphStyle(
            name='Code',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=10,
            spaceAfter=6,
            leftIndent=12,
            rightIndent=12,
            textColor=colors.HexColor('#333366')
        ))
        styles.add(ParagraphStyle(
            name='List',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leftIndent=12,
            rightIndent=12,
            textColor=colors.HexColor('#333366')
        ))
        styles.add(ParagraphStyle(
            name='ListItem',
            parent=styles['List'],
            fontSize=10,
            spaceAfter=6,
            leftIndent=12,
            rightIndent=12,
            textColor=colors.HexColor('#333366')
        ))
        styles.add(ParagraphStyle(
            name='Quote',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leftIndent=12,
            rightIndent=12,
            textColor=colors.HexColor('#333366')
        ))
        styles.add(ParagraphStyle(
            name='TableHeader',
            parent=styles['Heading3'],
            fontSize=10,
            textColor=colors.white,
            alignment=TA_CENTER,
            backgroundColor=colors.HexColor('#333366')
        ))
        styles.add(ParagraphStyle(
            name='TableCell',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            alignment=TA_LEFT,
            backgroundColor=colors.white
        ))
        styles.add(ParagraphStyle(
            name='TableCellCenter',
            parent=styles['TableCell'],
            alignment=TA_CENTER,
        ))
        styles.add(ParagraphStyle(
            name='TableCellRight',
            parent=styles['TableCell'],
            alignment=TA_RIGHT,
        ))
        styles.add(ParagraphStyle(
            name='TableCellLeft',
            parent=styles['TableCell'],
            alignment=TA_LEFT,
        ))  
        styles.add(ParagraphStyle(
            name='TableCellBold',
            parent=styles['TableCell'],
            fontSize=10,
            textColor=colors.black,
            alignment=TA_LEFT,
            backgroundColor=colors.white,
            fontName='Helvetica-Bold'
        ))
        styles.add(ParagraphStyle(
            name='TableCellBoldCenter',
            parent=styles['TableCellBold'],
            alignment=TA_CENTER,
        ))
        styles.add(ParagraphStyle(
            name='TableCellBoldRight',
            parent=styles['TableCellBold'],
            alignment=TA_RIGHT,
        ))
        styles.add(ParagraphStyle(
            name='TableCellBoldLeft',
            parent=styles['TableCellBold'],
            alignment=TA_LEFT,
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalic',
            parent=styles['TableCell'],
            fontSize=10,
            textColor=colors.black,
            alignment=TA_LEFT,
            backgroundColor=colors.white,
            fontName='Helvetica-Oblique'
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicCenter',
            parent=styles['TableCellItalic'],
            alignment=TA_CENTER,
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicRight',
            parent=styles['TableCellItalic'],
            alignment=TA_RIGHT,
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicLeft',
            parent=styles['TableCellItalic'],
            alignment=TA_LEFT,
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicBold',
            parent=styles['TableCellItalic'],
            fontSize=10,
            textColor=colors.black,
            alignment=TA_LEFT,
            backgroundColor=colors.white,
            fontName='Helvetica-BoldOblique'
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicBoldCenter',
            parent=styles['TableCellItalicBold'],
            alignment=TA_CENTER,
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicBoldRight',
            parent=styles['TableCellItalicBold'],
            alignment=TA_RIGHT,
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicBoldLeft',
            parent=styles['TableCellItalicBold'],
            alignment=TA_LEFT,
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicBoldLeft',
            parent=styles['TableCellItalicBold'],
            alignment=TA_LEFT,
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicBoldRight',
            parent=styles['TableCellItalicBold'],
            alignment=TA_RIGHT,
        ))
        styles.add(ParagraphStyle(
            name='TableCellItalicBoldCenter',
            parent=styles['TableCellItalicBold'],
            alignment=TA_CENTER,
        ))
        # Add more styles as needed for specific formatting
        elements = []

        # Add title
        elements.append(Paragraph(f"{symbol} {quarter_str} Earnings Call Report", styles['Title']))
        elements.append(Spacer(1, 12))

        # Add timestamp
        elements.append(Paragraph(f"Generated on: {timestamp}", styles['Normal']))
        elements.append(Spacer(1, 12))

        # Add sentiment chart
        if sentiment_chart:
            elements.append(Paragraph("Sentiment Trend During Call", styles['Heading2']))
            elements.append(Spacer(1, 12))
            elements.append(Image(sentiment_chart, width=6 * inch, height=3 * inch))
            elements.append(Spacer(1, 12))

        # Add speaker sentiment chart
        if speaker_chart:
            elements.append(Paragraph("Sentiment by Speaker", styles['Heading2']))
            elements.append(Spacer(1, 12))
            elements.append(Image(speaker_chart, width=6 * inch, height=3 * inch))
            elements.append(Spacer(1, 12))

        # Add metrics comparison chart
        if metrics_chart:
            elements.append(Paragraph("Quarterly Financial Metrics Comparison", styles['Heading2']))
            elements.append(Spacer(1, 12))
            elements.append(Image(metrics_chart, width=6 * inch, height=3 * inch))
            elements.append(Spacer(1, 12))

        # Add summaries
        elements.append(Paragraph("Financial Results", styles['Heading2']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(summary_data.get("metrics_summary", "No financial data available."), styles['Normal']))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph("Sentiment Analysis", styles['Heading2']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(summary_data.get("sentiment_summary", "No sentiment analysis available."), styles['Normal']))
        elements.append(Spacer(1, 12))

        if comparison_data:
            elements.append(Paragraph("Quarter-over-Quarter Comparison", styles['Heading2']))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(summary_data.get("comparison_summary", "No comparison data available."), styles['Normal']))
            elements.append(Spacer(1, 12))

        # Build the PDF
        doc.build(elements)
        return True

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return False

def generate_pdf_report(summary_data: Dict[str, Any], output_path: str, method: str = "reportlab") -> bool:
    """
    Generate a PDF report using the specified method.
    
    Args:
        summary_data: Comprehensive summary data
        output_path: Path to save the PDF
        method: PDF generation method ("reportlab" or "markdown")
        
    Returns:
        True if successful, False otherwise
    """
    # Validate input
    if not summary_data:
        print("Error: No summary data provided")
        return False
    
    if not output_path:
        print("Error: No output path provided")
        return False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return False
    
    # Generate PDF based on method
    if method.lower() == "reportlab":
        return generate_pdf_report_with_reportlab(summary_data, output_path)
    elif method.lower() == "markdown":
        # Generate markdown from summary data
        md_content = generate_markdown_summary(summary_data)
        return markdown_to_pdf(md_content, output_path)
    else:
        print(f"Error: Unsupported PDF generation method: {method}")
        return False


def main():
    """
    Test function to generate a PDF report
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PDF report from earnings call analysis")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol")
    parser.add_argument("--quarter", type=int, required=True, help="Quarter number (1-4)")
    parser.add_argument("--year", type=int, required=True, help="Year")
    parser.add_argument("--output", type=str, default=None, help="Output PDF path")
    parser.add_argument("--method", type=str, default="reportlab", choices=["reportlab", "markdown"], 
                        help="PDF generation method")
    
    args = parser.parse_args()
    
    # Generate file name if not provided
    if not args.output:
        output_dir = DATA_DIR / "reports"
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"{args.symbol.lower()}_q{args.quarter}_{args.year}_report.pdf")
    
    # Generate comprehensive summary
    summary_data = generate_comprehensive_summary(args.symbol, args.quarter, args.year)
    
    # Generate PDF report
    success = generate_pdf_report(summary_data, args.output, args.method)
    
    if success:
        print(f"PDF report generated successfully: {args.output}")
    else:
        print("Failed to generate PDF report")


if __name__ == "__main__":
    main()

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Global stylesheet variable
STYLES = None

def get_stylesheet():
    """
    Get the ReportLab stylesheet, initializing it only once.
    """
    global STYLES
    if STYLES is None:
        STYLES = getSampleStyleSheet()

        # Add custom styles
        STYLES.add(ParagraphStyle(name='Justify', parent=STYLES['Normal'], alignment=TA_JUSTIFY))
        STYLES.add(ParagraphStyle(name='Bullet', parent=STYLES['Normal'], leftIndent=36))
        STYLES.add(ParagraphStyle(name='Code', parent=STYLES['Normal'], fontName='Courier', fontSize=10, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='List', parent=STYLES['Normal'], fontSize=10, spaceAfter=6, leftIndent=12, rightIndent=12, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='ListItem', parent=STYLES['List'], fontSize=10, spaceAfter=6, leftIndent=12, rightIndent=12, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Quote', parent=STYLES['Normal'], fontSize=10, spaceAfter=6, leftIndent=12, rightIndent=12, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='H1', parent=STYLES['Heading1'], fontSize=24, leading=28, spaceAfter=12, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='H2', parent=STYLES['Heading2'], fontSize=18, leading=22, spaceAfter=8, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='H3', parent=STYLES['Heading3'], fontSize=14, leading=18, spaceAfter=6, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Normal', parent=STYLES['Normal'], fontSize=10, leading=12, spaceAfter=6, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Italic', parent=STYLES['Normal'], fontName='Times-Italic', textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Bold', parent=STYLES['Normal'], fontName='Times-Bold', textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Title', parent=STYLES['Title'], fontName='Times-Bold', fontSize=28, leading=32, alignment=TA_CENTER, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Heading1', parent=STYLES['Heading1'], fontName='Times-Bold', fontSize=20, leading=24, spaceAfter=12, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Heading2', parent=STYLES['Heading2'], fontName='Times-Bold', fontSize=16, leading=20, spaceAfter=8, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Heading3', parent=STYLES['Heading3'], fontName='Times-Bold', fontSize=12, leading=16, spaceAfter=6, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Heading4', parent=STYLES['Heading4'], fontName='Times-Bold', fontSize=10, leading=14, spaceAfter=4, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Heading5', parent=STYLES['Heading5'], fontName='Times-Bold', fontSize=8, leading=12, spaceAfter=2, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='Heading6', parent=STYLES['Heading6'], fontName='Times-Bold', fontSize=6, leading=10, spaceAfter=0, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='h1', parent=STYLES['Heading1'], fontSize=24, leading=28, spaceAfter=12, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='h2', parent=STYLES['Heading2'], fontSize=18, leading=22, spaceAfter=8, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='h3', parent=STYLES['Heading3'], fontSize=14, leading=18, spaceAfter=6, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='h4', parent=STYLES['Heading4'], fontSize=10, leading=14, spaceAfter=4, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='h5', parent=STYLES['Heading5'], fontSize=8, leading=12, spaceAfter=2, textColor=colors.HexColor('#333366')))
        STYLES.add(ParagraphStyle(name='h6', parent=STYLES['Heading6'], fontSize=6, leading=10, spaceAfter=0, textColor=colors.HexColor('#333366')))
    return STYLES

def generate_pdf_report(summary_data: Dict[str, Any], output_path: str, method: str = "reportlab") -> bool:
    """
    Generate a PDF report using ReportLab.
    """
    try:
        # Get stylesheet
        styles = get_stylesheet()

        # Create document
        doc = SimpleDocTemplate(output_path, pagesize=letter)

        # Story elements
        story = []

        # Title
        title = summary_data.get("title", "Earnings Call Report")
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 0.2 * inch))

        # Executive Summary
        executive_summary = summary_data.get("executive_summary", "No summary available.")
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        story.append(Paragraph(executive_summary, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Key Points
        key_points = summary_data.get("key_points", [])
        story.append(Paragraph("Key Points", styles['Heading1']))
        for point in key_points:
            story.append(Paragraph(point, styles['Bullet']))
        story.append(Spacer(1, 0.2 * inch))

        # Build the document
        doc.build(story)

        return True

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return False