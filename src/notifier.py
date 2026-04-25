"""
notifier.py
Email alert system for Suspicious Activity Detection.
Sends a real email with threat details and an attached snapshot
whenever a high-confidence alert fires.

Configuration is loaded from alert_config.json (auto-created on first run).
Uses Gmail SMTP with an App Password — works without 2FA complications.
"""

import smtplib
import json
import os
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE  = os.path.join(SCRIPT_DIR, 'alert_config.json')

_DEFAULT_CONFIG = {
    "enabled":        False,
    "smtp_host":      "smtp.gmail.com",
    "smtp_port":      465,
    "use_ssl":        True,
    "recipient_email": "",
    "min_interval_s": 30
}

# Sender credentials are read exclusively from environment variables —
# never stored in the config file or shown in the UI.
_SENDER_EMAIL    = os.environ.get("SURV_SENDER_EMAIL", "")
_SENDER_PASSWORD = os.environ.get("SURV_SENDER_PASSWORD", "")

_last_sent_time = 0
_config_cache   = None
_send_lock      = threading.Lock()


def load_config() -> dict:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            _config_cache = {**_DEFAULT_CONFIG, **cfg}
            return _config_cache
        except Exception:
            pass
    _config_cache = dict(_DEFAULT_CONFIG)
    return _config_cache


def save_config(cfg: dict):
    global _config_cache
    _config_cache = cfg
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[NOTIFIER] Could not save config: {e}")


def send_alert(activity_type: str = "SUSPICIOUS",
               confidence: float = 0.0,
               snapshot_path: str = None,
               camera_id: str = "CAM-1"):
    """
    Send a threat alert email. Called from a background thread — all
    SMTP work happens inside another daemon thread so it never blocks
    the video loop.
    """
    threading.Thread(
        target=_send_worker,
        args=(activity_type, confidence, snapshot_path, camera_id),
        daemon=True
    ).start()


def _send_worker(activity_type: str, confidence: float, snapshot_path: str, camera_id: str = "CAM-1"):
    global _last_sent_time

    cfg = load_config()

    if not cfg.get("enabled"):
        return
    if not _SENDER_EMAIL or not _SENDER_PASSWORD:
        print("[NOTIFIER] Sender credentials missing. Set SURV_SENDER_EMAIL and SURV_SENDER_PASSWORD env vars.")
        return
    if not cfg.get("recipient_email"):
        print("[NOTIFIER] Recipient email not configured. Open Detection Settings → Email Alerts.")
        return

    with _send_lock:
        import time
        now = time.time()
        if now - _last_sent_time < cfg.get("min_interval_s", 30):
            return
        _last_sent_time = now

    ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf_pct = f"{confidence * 100:.1f}%"
    subject  = f"\U0001f6a8 [{camera_id}] Suspicious Activity — {activity_type}"

    html_body = f"""
    <html><body style="font-family:Arial,sans-serif;background:#f5f5f5;padding:20px;">
      <div style="max-width:560px;margin:auto;background:white;border-radius:8px;
                  border-left:6px solid #e53935;padding:24px;">
        <h2 style="color:#e53935;margin-top:0;">&#9888; Suspicious Activity Alert</h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr><td style="padding:6px 0;color:#555;width:140px;">Camera</td>
              <td style="padding:6px 0;font-weight:bold;">{camera_id}</td></tr>
          <tr><td style="padding:6px 0;color:#555;">Activity Type</td>
              <td style="padding:6px 0;font-weight:bold;">{activity_type}</td></tr>
          <tr><td style="padding:6px 0;color:#555;">Confidence</td>
              <td style="padding:6px 0;font-weight:bold;color:#e53935;">{conf_pct}</td></tr>
          <tr><td style="padding:6px 0;color:#555;">Timestamp</td>
              <td style="padding:6px 0;">{ts}</td></tr>
        </table>
        {"<p style='margin-top:16px;color:#555;font-size:13px;'>A snapshot of the detection frame is attached.</p>" if snapshot_path else ""}
        <hr style="margin:20px 0;border:none;border-top:1px solid #eee;">
        <p style="color:#aaa;font-size:12px;margin:0;">
          Hybrid AI Surveillance System &mdash; automated alert
        </p>
      </div>
    </body></html>
    """

    try:
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"]    = _SENDER_EMAIL
        msg["To"]      = cfg["recipient_email"]
        msg.attach(MIMEText(html_body, "html"))

        if snapshot_path and os.path.exists(snapshot_path):
            with open(snapshot_path, "rb") as img_f:
                img_data = img_f.read()
            img_part = MIMEImage(img_data, name=os.path.basename(snapshot_path))
            img_part.add_header("Content-Disposition", "attachment",
                                 filename=os.path.basename(snapshot_path))
            msg.attach(img_part)

        use_ssl = cfg.get("use_ssl", True)
        port    = cfg.get("smtp_port", 465)
        host    = cfg["smtp_host"]

        if use_ssl:
            with smtplib.SMTP_SSL(host, port) as server:
                server.login(_SENDER_EMAIL, _SENDER_PASSWORD)
                server.sendmail(_SENDER_EMAIL, cfg["recipient_email"], msg.as_string())
        else:
            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls()
                server.login(_SENDER_EMAIL, _SENDER_PASSWORD)
                server.sendmail(_SENDER_EMAIL, cfg["recipient_email"], msg.as_string())

        print(f"[NOTIFIER] Alert sent to {cfg['recipient_email']} ({activity_type})")

    except smtplib.SMTPAuthenticationError:
        print("[NOTIFIER] Auth failed. Use a Gmail App Password, not your account password.")
        print("           Guide: myaccount.google.com/apppasswords")
    except Exception as e:
        print(f"[NOTIFIER] Email send failed: {e}")
