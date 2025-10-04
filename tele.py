from telethon.sync import TelegramClient
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Telegram API credentials
api_id = 20938512
api_hash = '464421f40d609e18440a7796c4b4ce80'
phone = '+96176109211'

# Firestore initialization
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
district=next((text for text in a.split() if text in lebanese_places),None)
# Connect to Telegram
client = TelegramClient('session_name', api_id, api_hash)
client.connect()

if not client.is_user_authorized():
    client.send_code_request(phone)
    client.sign_in(phone, input('Enter the code: '))

# Telegram group/channel username or link
group_username = 'https://t.me/AjaNews'

entity = client.get_entity(group_username)

from telethon.tl.functions.messages import GetHistoryRequest

history = client(GetHistoryRequest(
    peer=entity,
    limit=100,  # number of messages to fetch
    offset_date=None,
    offset_id=0,
    max_id=0,
    min_id=0,
    add_offset=0,
    hash=0
))

# Just pick one message that has text (skip empty)
one_message = None
for message in history.messages:
    if message.message and message.message.strip():
        one_message = message.message.strip()
        break

if one_message is None:
    print("No suitable message found to add.")
else:
    print(f"Adding this message to Firestore:\n{one_message[:100]}...")

    # Prepare Firestore document - adjust fields if needed to match your schema
    doc_data = {
        "article": one_message,
        "district": None,  # no district info from Telegram, set to None or default
        "timestamp": datetime.now()
    }

    # Add document to 'articles' collection (the same one your FastAPI app reads from)
    added_doc_ref = db.collection("articles").add(doc_data)
    print(f"Message added to Firestore with ID: {added_doc_ref[1].id}")
