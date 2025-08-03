import pandas as pd
import yagmail
from datetime import datetime

def summarize_and_send_email():
    try:
        df = pd.read_csv("detections.csv")

        gender_counts = df["gender"].value_counts().to_dict()
        emotion_counts = df["emotion"].value_counts().to_dict()

        summary = f"""
        📊 Daily Detection Summary - {datetime.now().strftime('%Y-%m-%d')}

        👤 Genders:
        {gender_counts}

        😃 Emotions:
        {emotion_counts}

        📎 Attached: detections.csv
        """

        yag = yagmail.SMTP("titusjerwin@gmail.com", "lzlk wigl dact rvkp")  # Use App Password
        yag.send(
            to="",
            subject="📩 Daily Age-Gender-Emotion Report",
            contents=summary,
            attachments="detections.csv"
        )
        print("✅ Email sent successfully!")

    except Exception as e:
        print(f"❌ Error sending email: {e}")

if __name__ == "__main__":
    summarize_and_send_email()
