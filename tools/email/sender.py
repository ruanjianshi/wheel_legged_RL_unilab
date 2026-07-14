"""SMTP 邮件发送器"""

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from tools.email.config import SmtpConfig


def send_email(
    to: str,
    subject: str,
    body: str,
    body_html: str | None = None,
    cfg: SmtpConfig | None = None,
) -> bool:
    cfg = cfg or SmtpConfig()
    msg = MIMEMultipart("alternative")
    msg["From"] = cfg.from_addr
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    if body_html:
        msg.attach(MIMEText(body_html, "html", "utf-8"))

    try:
        if cfg.use_ssl:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(cfg.host, cfg.port, context=ctx) as s:
                s.login(cfg.user, cfg.password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(cfg.host, cfg.port) as s:
                s.starttls()
                s.login(cfg.user, cfg.password)
                s.send_message(msg)
        return True
    except smtplib.SMTPException as e:
        print(f"[email] 发送失败: {e}")
        return False
