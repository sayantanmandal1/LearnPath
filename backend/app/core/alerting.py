"""
Automated alerting system for system failures
"""
import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

import httpx
import structlog
from app.core.monitoring import Alert, AlertSeverity, AlertType, system_monitor
from app.core.config import settings

logger = structlog.get_logger()


class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DISCORD = "discord"


@dataclass
class NotificationConfig:
    """Notification configuration"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = None
    severity_filter: Set[AlertSeverity] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if self.severity_filter is None:
            self.severity_filter = {AlertSeverity.HIGH, AlertSeverity.CRITICAL}


class AlertingSystem:
    """Automated alerting system"""
    
    def __init__(self):
        self.notification_configs: List[NotificationConfig] = []
        self.alert_history: Dict[str, datetime] = {}
        self.rate_limit_window = 300  # 5 minutes
        self.max_alerts_per_window = 10
        self.escalation_rules: Dict[AlertSeverity, int] = {
            AlertSeverity.LOW: 3600,      # 1 hour
            AlertSeverity.MEDIUM: 1800,   # 30 minutes
            AlertSeverity.HIGH: 600,      # 10 minutes
            AlertSeverity.CRITICAL: 60,   # 1 minute
        }
        
        # Setup default configurations
        self.setup_default_configs()
        
        # Register with monitoring system
        system_monitor.add_alert_handler(self.handle_alert)
    
    def setup_default_configs(self):
        """Setup default notification configurations"""
        # Email notifications for high and critical alerts
        if hasattr(settings, 'SMTP_HOST') and settings.SMTP_HOST:
            self.add_notification_config(
                NotificationConfig(
                    channel=NotificationChannel.EMAIL,
                    enabled=True,
                    config={
                        "smtp_host": getattr(settings, 'SMTP_HOST', 'localhost'),
                        "smtp_port": getattr(settings, 'SMTP_PORT', 587),
                        "smtp_username": getattr(settings, 'SMTP_USERNAME', ''),
                        "smtp_password": getattr(settings, 'SMTP_PASSWORD', ''),
                        "from_email": getattr(settings, 'ALERT_FROM_EMAIL', 'alerts@aicareer.com'),
                        "to_emails": getattr(settings, 'ALERT_TO_EMAILS', ['admin@aicareer.com']),
                        "use_tls": getattr(settings, 'SMTP_USE_TLS', True),
                    },
                    severity_filter={AlertSeverity.HIGH, AlertSeverity.CRITICAL}
                )
            )
        
        # Slack notifications
        if hasattr(settings, 'SLACK_WEBHOOK_URL') and settings.SLACK_WEBHOOK_URL:
            self.add_notification_config(
                NotificationConfig(
                    channel=NotificationChannel.SLACK,
                    enabled=True,
                    config={
                        "webhook_url": settings.SLACK_WEBHOOK_URL,
                        "channel": getattr(settings, 'SLACK_CHANNEL', '#alerts'),
                        "username": getattr(settings, 'SLACK_USERNAME', 'AI Career Bot'),
                    },
                    severity_filter={AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL}
                )
            )
        
        # Generic webhook notifications
        if hasattr(settings, 'ALERT_WEBHOOK_URL') and settings.ALERT_WEBHOOK_URL:
            self.add_notification_config(
                NotificationConfig(
                    channel=NotificationChannel.WEBHOOK,
                    enabled=True,
                    config={
                        "webhook_url": settings.ALERT_WEBHOOK_URL,
                        "headers": getattr(settings, 'ALERT_WEBHOOK_HEADERS', {}),
                        "timeout": 30,
                    },
                    severity_filter={AlertSeverity.HIGH, AlertSeverity.CRITICAL}
                )
            )
    
    def add_notification_config(self, config: NotificationConfig):
        """Add notification configuration"""
        self.notification_configs.append(config)
        logger.info(f"Added notification config: {config.channel.value}")
    
    def handle_alert(self, alert: Alert):
        """Handle incoming alert"""
        try:
            # Check rate limiting
            if self.is_rate_limited(alert):
                logger.warning(f"Alert rate limited: {alert.id}")
                return
            
            # Record alert
            self.alert_history[alert.id] = alert.timestamp
            
            # Send notifications
            asyncio.create_task(self.send_notifications(alert))
            
        except Exception as e:
            logger.error(f"Error handling alert {alert.id}: {e}")
    
    def is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert should be rate limited"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.rate_limit_window)
        
        # Count recent alerts
        recent_alerts = sum(
            1 for timestamp in self.alert_history.values()
            if timestamp > window_start
        )
        
        return recent_alerts >= self.max_alerts_per_window
    
    async def send_notifications(self, alert: Alert):
        """Send notifications for alert"""
        tasks = []
        
        for config in self.notification_configs:
            if not config.enabled:
                continue
            
            # Check severity filter
            if config.severity_filter and alert.severity not in config.severity_filter:
                continue
            
            # Create notification task
            task = asyncio.create_task(
                self.send_notification(alert, config)
            )
            tasks.append(task)
        
        # Wait for all notifications to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    config = self.notification_configs[i]
                    logger.error(
                        f"Notification failed: {config.channel.value}",
                        error=str(result),
                        alert_id=alert.id
                    )
    
    async def send_notification(self, alert: Alert, config: NotificationConfig):
        """Send notification via specific channel"""
        try:
            if config.channel == NotificationChannel.EMAIL:
                await self.send_email_notification(alert, config)
            elif config.channel == NotificationChannel.SLACK:
                await self.send_slack_notification(alert, config)
            elif config.channel == NotificationChannel.WEBHOOK:
                await self.send_webhook_notification(alert, config)
            elif config.channel == NotificationChannel.DISCORD:
                await self.send_discord_notification(alert, config)
            else:
                logger.warning(f"Unsupported notification channel: {config.channel}")
                
        except Exception as e:
            logger.error(f"Failed to send {config.channel.value} notification: {e}")
            raise
    
    async def send_email_notification(self, alert: Alert, config: NotificationConfig):
        """Send email notification"""
        smtp_config = config.config
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_config['from_email']
        msg['To'] = ', '.join(smtp_config['to_emails'])
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        # Create email body
        body = self.create_email_body(alert)
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        def send_email():
            with smtplib.SMTP(smtp_config['smtp_host'], smtp_config['smtp_port']) as server:
                if smtp_config.get('use_tls', True):
                    server.starttls()
                
                if smtp_config.get('smtp_username'):
                    server.login(smtp_config['smtp_username'], smtp_config['smtp_password'])
                
                server.send_message(msg)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_email)
        
        logger.info(f"Email notification sent for alert: {alert.id}")
    
    async def send_slack_notification(self, alert: Alert, config: NotificationConfig):
        """Send Slack notification"""
        slack_config = config.config
        
        # Create Slack message
        color_map = {
            AlertSeverity.LOW: "#36a64f",      # Green
            AlertSeverity.MEDIUM: "#ff9500",   # Orange
            AlertSeverity.HIGH: "#ff0000",     # Red
            AlertSeverity.CRITICAL: "#8B0000", # Dark Red
        }
        
        payload = {
            "channel": slack_config.get('channel', '#alerts'),
            "username": slack_config.get('username', 'AI Career Bot'),
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#ff0000"),
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Service",
                            "value": alert.service or "System",
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "short": True
                        },
                        {
                            "title": "Alert ID",
                            "value": alert.id,
                            "short": True
                        }
                    ],
                    "footer": "AI Career Recommender",
                    "ts": int(alert.timestamp.timestamp())
                }
            ]
        }
        
        # Add recovery suggestions if available
        if alert.recovery_suggestions:
            payload["attachments"][0]["fields"].append({
                "title": "Recovery Suggestions",
                "value": "\n".join(f"• {suggestion}" for suggestion in alert.recovery_suggestions),
                "short": False
            })
        
        # Send to Slack
        async with httpx.AsyncClient() as client:
            response = await client.post(
                slack_config['webhook_url'],
                json=payload,
                timeout=30
            )
            response.raise_for_status()
        
        logger.info(f"Slack notification sent for alert: {alert.id}")
    
    async def send_webhook_notification(self, alert: Alert, config: NotificationConfig):
        """Send webhook notification"""
        webhook_config = config.config
        
        # Create webhook payload
        payload = {
            "alert": alert.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "ai-career-recommender"
        }
        
        # Send webhook
        headers = webhook_config.get('headers', {})
        headers.setdefault('Content-Type', 'application/json')
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_config['webhook_url'],
                json=payload,
                headers=headers,
                timeout=webhook_config.get('timeout', 30)
            )
            response.raise_for_status()
        
        logger.info(f"Webhook notification sent for alert: {alert.id}")
    
    async def send_discord_notification(self, alert: Alert, config: NotificationConfig):
        """Send Discord notification"""
        discord_config = config.config
        
        # Create Discord embed
        color_map = {
            AlertSeverity.LOW: 0x36a64f,      # Green
            AlertSeverity.MEDIUM: 0xff9500,   # Orange
            AlertSeverity.HIGH: 0xff0000,     # Red
            AlertSeverity.CRITICAL: 0x8B0000, # Dark Red
        }
        
        embed = {
            "title": f"[{alert.severity.value.upper()}] {alert.title}",
            "description": alert.description,
            "color": color_map.get(alert.severity, 0xff0000),
            "timestamp": alert.timestamp.isoformat(),
            "fields": [
                {
                    "name": "Service",
                    "value": alert.service or "System",
                    "inline": True
                },
                {
                    "name": "Alert ID",
                    "value": alert.id,
                    "inline": True
                }
            ],
            "footer": {
                "text": "AI Career Recommender"
            }
        }
        
        # Add recovery suggestions
        if alert.recovery_suggestions:
            embed["fields"].append({
                "name": "Recovery Suggestions",
                "value": "\n".join(f"• {suggestion}" for suggestion in alert.recovery_suggestions),
                "inline": False
            })
        
        payload = {
            "embeds": [embed]
        }
        
        # Send to Discord
        async with httpx.AsyncClient() as client:
            response = await client.post(
                discord_config['webhook_url'],
                json=payload,
                timeout=30
            )
            response.raise_for_status()
        
        logger.info(f"Discord notification sent for alert: {alert.id}")
    
    def create_email_body(self, alert: Alert) -> str:
        """Create HTML email body"""
        severity_colors = {
            AlertSeverity.LOW: "#28a745",
            AlertSeverity.MEDIUM: "#ffc107",
            AlertSeverity.HIGH: "#fd7e14",
            AlertSeverity.CRITICAL: "#dc3545",
        }
        
        color = severity_colors.get(alert.severity, "#dc3545")
        
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="background-color: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0;">
                    <h1 style="margin: 0; font-size: 24px;">[{alert.severity.value.upper()}] System Alert</h1>
                </div>
                
                <div style="padding: 20px;">
                    <h2 style="color: #333; margin-top: 0;">{alert.title}</h2>
                    <p style="color: #666; font-size: 16px; line-height: 1.5;">{alert.description}</p>
                    
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold; color: #333;">Alert ID:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">{alert.id}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold; color: #333;">Service:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">{alert.service or 'System'}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold; color: #333;">Timestamp:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold; color: #333;">Severity:</td>
                            <td style="padding: 8px; border-bottom: 1px solid #eee;">
                                <span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">
                                    {alert.severity.value.upper()}
                                </span>
                            </td>
                        </tr>
                    </table>
        """
        
        # Add recovery suggestions
        if alert.recovery_suggestions:
            html += """
                    <div style="background-color: #e7f3ff; border-left: 4px solid #007bff; padding: 15px; margin: 20px 0;">
                        <h3 style="color: #007bff; margin-top: 0;">Recovery Suggestions:</h3>
                        <ul style="color: #333; margin: 0;">
            """
            for suggestion in alert.recovery_suggestions:
                html += f"<li style='margin-bottom: 5px;'>{suggestion}</li>"
            html += """
                        </ul>
                    </div>
            """
        
        # Add metadata if available
        if alert.metadata:
            html += """
                    <div style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px; padding: 15px; margin: 20px 0;">
                        <h3 style="color: #333; margin-top: 0;">Additional Details:</h3>
                        <pre style="background-color: white; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 12px;">
            """
            html += json.dumps(alert.metadata, indent=2)
            html += """
                        </pre>
                    </div>
            """
        
        html += """
                </div>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 0 0 8px 8px; text-align: center; color: #666; font-size: 12px;">
                    This alert was generated by AI Career Recommender monitoring system.
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    async def test_notifications(self, test_alert: Optional[Alert] = None):
        """Test all notification channels"""
        if not test_alert:
            test_alert = Alert(
                id="test_alert_" + str(int(datetime.utcnow().timestamp())),
                type=AlertType.SECURITY_INCIDENT,
                severity=AlertSeverity.MEDIUM,
                title="Test Alert",
                description="This is a test alert to verify notification channels are working correctly.",
                timestamp=datetime.utcnow(),
                service="test_service",
                metadata={"test": True},
                recovery_suggestions=["This is a test alert", "No action required"]
            )
        
        await self.send_notifications(test_alert)
        logger.info("Test notifications sent")


# Global alerting system instance
alerting_system = AlertingSystem()