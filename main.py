#C:\miniconda\Conda_install\AI\Auto_Tagging\auto_tagging
import os
import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

df = pd.read_csv("C:\miniconda\Conda_install\AI\Auto_Tagging\Ticket.txt")

CANDIDATE_TAGS = [
    "Login", "Payment", "Bug", "Crash", "UI", 
    "Feature Request", "Account", "Mobile", "Security"
]

def zero_shot_tagging(ticket):
    prompt = f"""
You are a support ticket classifier.
Ticket: "{ticket}"
Choose the 3 most relevant tags from the following list:
{', '.join(CANDIDATE_TAGS)}
Tags:"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"


few_shot_examples = """
Message: "App crashes on startup."
Tags: Crash, Bug, Mobile

Message: "I can't log in to my account."
Tags: Login, Account

Message: "Please add a dark mode feature."
Tags: Feature Request, UI

Message: "I was charged twice for one order."
Tags: Payment, Bug

Message: "I forgot my password."
Tags: Login, Account

Message: "My app doesn't open on iPhone."
Tags: Crash, Mobile, Bug

Message: "My screen layout is broken on desktop."
Tags: UI, Bug

Message: "I need to change my email."
Tags: Account, Security
"""

def few_shot_tagging(ticket):
    prompt = few_shot_examples + f'\nMessage: "{ticket}"\nTags:'
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

print("Running Zero-Shot Tagging...")
df['Tags_Zero_Shot'] = df['Ticket'].apply(zero_shot_tagging)

print("Running Few-Shot Tagging...")
df['Tags_Few_Shot'] = df['Ticket'].apply(few_shot_tagging)


df.to_csv("Tagged_Tickets.csv", index=False)
print("Tagging complete! Results saved to Tagged_Tickets.csv.")
