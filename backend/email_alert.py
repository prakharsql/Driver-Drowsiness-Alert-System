import smtplib
from email.mime.text import MIMEText

SENDER_EMAIL = "your_email@gmail.com"
APP_PASSWORD = "your_app_password"
OWNER_EMAIL = "owner_email@gmail.com"

def send_email(message):
    msg = MIMEText(message)
    msg["Subject"] = "🚨 Driver Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = OWNER_EMAIL

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.send_message(msg)
    server.quit()

    print("📧 Email sent")
