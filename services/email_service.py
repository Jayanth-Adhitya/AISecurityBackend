import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
from pathlib import Path
import asyncio
from datetime import datetime
import aiosmtplib
from jinja2 import Template
import os
from config import settings

class EmailService:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_user)
        self.alert_recipients = os.getenv("ALERT_RECIPIENTS", "").split(",")
        
        # Email templates
        self.anomaly_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .alert-box { 
                    background-color: #ff4444; 
                    color: white; 
                    padding: 15px; 
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .details { 
                    background-color: #f0f0f0; 
                    padding: 15px; 
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .severity-high { color: #ff0000; font-weight: bold; }
                .severity-medium { color: #ff9900; font-weight: bold; }
                .severity-low { color: #ffcc00; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                .image-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
                .image-container { text-align: center; }
                .image-container img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <div class="alert-box">
                <h2>⚠️ ANOMALY DETECTED - {{ anomaly_type }}</h2>
                <p>Time: {{ detection_time }}</p>
                <p>Video: {{ video_name }}</p>
            </div>
            
            <div class="details">
                <h3>Anomaly Details:</h3>
                <table>
                    <tr>
                        <th>Property</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Type</td>
                        <td>{{ anomaly_type }}</td>
                    </tr>
                    <tr>
                        <td>Severity</td>
                        <td class="severity-{{ severity }}">{{ severity }}</td>
                    </tr>
                    <tr>
                        <td>Confidence</td>
                        <td>{{ confidence }}%</td>
                    </tr>
                    <tr>
                        <td>Location in Video</td>
                        <td>{{ timestamp }}s (Frame: {{ frame_number }})</td>
                    </tr>
                    <tr>
                        <td>Description</td>
                        <td>{{ description }}</td>
                    </tr>
                </table>
            </div>
            
            {% if additional_info %}
            <div class="details">
                <h3>Additional Information:</h3>
                <ul>
                {% for info in additional_info %}
                    <li>{{ info }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            <div class="details">
                <p><strong>Action Required:</strong> Please review the video footage and take appropriate action if necessary.</p>
                <p><a href="{{ dashboard_url }}">View in Dashboard</a></p>
            </div>
        </body>
        </html>
        """
        
        self.daily_summary_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                .header { background-color: #4CAF50; color: white; padding: 20px; }
                .summary-box { background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Daily Traffic Analysis Summary</h2>
                <p>Date: {{ date }}</p>
            </div>
            
            <div class="summary-box">
                <h3>Anomaly Summary</h3>
                <table>
                    <tr>
                        <th>Anomaly Type</th>
                        <th>Count</th>
                        <th>Severity</th>
                    </tr>
                    {% for anomaly in anomalies %}
                    <tr>
                        <td>{{ anomaly.type }}</td>
                        <td>{{ anomaly.count }}</td>
                        <td>{{ anomaly.severity }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="summary-box">
                <h3>Traffic Statistics</h3>
                <ul>
                    <li>Total Vehicles: {{ total_vehicles }}</li>
                    <li>Total Pedestrians: {{ total_pedestrians }}</li>
                    <li>Peak Hour: {{ peak_hour }}</li>
                    <li>Videos Processed: {{ videos_processed }}</li>
                </ul>
            </div>
        </body>
        </html>
        """
    
    async def send_anomaly_alert(
        self,
        recipient_email: str,
        anomaly_type: str,
        severity: str,  # high, medium, low
        confidence: float,
        video_id: int,
        video_name: str,
        frame_number: int,
        timestamp: float,
        description: str,
        luggage_description: str = "Unknown",
        frame_images: Optional[List[Path]] = None,
        additional_info: Optional[List[str]] = None,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """
        Send security alert email for abandoned luggage with attached images.

        Args:
            recipient_email: Primary recipient email (session-based)
            luggage_description: Description of the luggage (color, type)
            ... (other params)
        """
        try:
            if not self.smtp_user or not self.smtp_password:
                print("Email credentials not configured")
                return False

            # Use recipient_email as primary recipient
            if recipient_email:
                recipients = [recipient_email]
            else:
                recipients = recipients or self.alert_recipients

            if not recipients:
                print("No email recipients configured")
                return False
            
            # Prepare template data with luggage info
            template = Template(self.anomaly_template)
            html_content = template.render(
                anomaly_type=anomaly_type.replace('_', ' ').title(),
                severity=severity.lower(),
                confidence=round(confidence * 100, 1),
                video_name=video_name,
                detection_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                timestamp=round(timestamp, 1),
                frame_number=frame_number,
                description=description,
                luggage_description=luggage_description,
                additional_info=additional_info or [],
                dashboard_url=f"http://localhost:3000/videos/{video_id}"
            )

            # Create message
            msg = MIMEMultipart('related')
            msg['Subject'] = f"🚨 SECURITY ALERT: {anomaly_type.replace('_', ' ').title()}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            
            # Attach HTML content
            msg.attach(MIMEText(html_content, 'html'))
            
            # Attach frame images if provided
            if frame_images:
                for idx, image_path in enumerate(frame_images[:4]):  # Limit to 4 images
                    if image_path.exists():
                        with open(image_path, 'rb') as f:
                            img = MIMEImage(f.read())
                            img.add_header('Content-ID', f'<frame{idx}>')
                            img.add_header('Content-Disposition', 'inline', 
                                         filename=f'frame_{idx}.jpg')
                            msg.attach(img)
            
            # Send email asynchronously
            await self._send_email_async(msg, recipients)
            
            print(f"Anomaly alert sent successfully to {recipients}")
            return True
            
        except Exception as e:
            print(f"Failed to send anomaly alert: {e}")
            return False
    
    async def send_daily_summary(
        self,
        date: str,
        anomalies: List[Dict[str, Any]],
        total_vehicles: int,
        total_pedestrians: int,
        peak_hour: str,
        videos_processed: int,
        recipients: Optional[List[str]] = None
    ) -> bool:
        """Send daily summary email."""
        try:
            recipients = recipients or self.alert_recipients
            if not recipients:
                return False
            
            template = Template(self.daily_summary_template)
            html_content = template.render(
                date=date,
                anomalies=anomalies,
                total_vehicles=total_vehicles,
                total_pedestrians=total_pedestrians,
                peak_hour=peak_hour,
                videos_processed=videos_processed
            )
            
            msg = MIMEMultipart()
            msg['Subject'] = f"Daily Traffic Analysis Summary - {date}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            msg.attach(MIMEText(html_content, 'html'))
            
            await self._send_email_async(msg, recipients)
            
            print(f"Daily summary sent successfully to {recipients}")
            return True
            
        except Exception as e:
            print(f"Failed to send daily summary: {e}")
            return False
    
    async def _send_email_async(self, message: MIMEMultipart, recipients: List[str]):
        """Send email asynchronously using aiosmtplib."""
        try:
            # Use aiosmtplib for async email sending
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                start_tls=True
            )
        except Exception as e:
            # Fallback to synchronous sending
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(message)