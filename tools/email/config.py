"""邮件配置"""
import os
from dataclasses import dataclass, field

@dataclass
class SmtpConfig:
    host: str = os.environ.get("UNILAB_SMTP_HOST", "smtp.2925.com")
    port: int = int(os.environ.get("UNILAB_SMTP_PORT", "465"))
    user: str = os.environ.get("UNILAB_SMTP_USER", "")
    password: str = os.environ.get("UNILAB_SMTP_PASS", "")
    use_ssl: bool = True
    from_addr: str = ""

    def __post_init__(self):
        if self.user and "@" not in self.user:
            self.user = f"{self.user}@2925.com"
        if not self.from_addr:
            self.from_addr = self.user