import smtplib
from email.mime.text import MIMEText

SENDER_EMAIL = "pdprakhar03@gmail.com"
APP_PASSWORD = "jqnx nbiv ytmb vzee"
OWNER_EMAIL = "pdprakhar03@gmail.com"

def send_email(message):
    msg = MIMEText(message)
    msg["Subject"] = "🚨 Driver Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = OWNER_EMAIL

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)

    print("📧 Email sent")
