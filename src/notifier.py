"""
Email Alert Module for Suspicious Activity Detection

Sends email alerts when suspicious activity is detected.
Configure credentials via environment variables:
  - ALERT_SENDER_EMAIL
  - ALERT_APP_PASSWORD
  - ALERT_RECIPIENT_EMAIL
"""

import smtplib
import os
from email.message import EmailMessage


def send_alert(subject="Suspicious Activity Detection", body="Suspicious Activity Detected"):
    """Send an email alert. Returns True on success, False on failure."""
    SENDER_EMAIL = os.environ.get("ALERT_SENDER_EMAIL", "")
    APP_PASSWORD = os.environ.get("ALERT_APP_PASSWORD", "")
    RECIPIENT_EMAIL = os.environ.get("ALERT_RECIPIENT_EMAIL", "")

    if not SENDER_EMAIL or not APP_PASSWORD or not RECIPIENT_EMAIL:
        print("[mail.py] Warning: Email credentials not configured.")
        print("  Set ALERT_SENDER_EMAIL, ALERT_APP_PASSWORD, and ALERT_RECIPIENT_EMAIL environment variables.")
        return False

    try:
        print("Mail Start")
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg.set_content(body)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
            print("Mail sent successfully")
        return True
    except Exception as e:
        print(f"[mail.py] Error sending email: {e}")
        return False


if __name__ == "__main__":
    send_alert()
