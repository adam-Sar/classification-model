import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from classifier import classify_article
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import logging
from districts import lebanese_places
from deep_translator import GoogleTranslator
from langdetect import detect
translator = GoogleTranslator(source="auto", target="en")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Firestore
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Background task reference
background_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application")
    global background_task
    background_task = asyncio.create_task(process_articles())
    yield
    # Shutdown
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutting down application")

app = FastAPI(lifespan=lifespan)

async def check_new_articles():
    try:
        # Query for all documents in articles collection
        articles_ref = db.collection('articles')
        docs = articles_ref.get()
        
        new_articles = []
        for doc in docs:
            # Check if document has already been processed
            processed_doc = db.collection('processed_articles').document(doc.id).get()
            if processed_doc.exists:
                continue 
            doc_data = doc.to_dict() 
            #get article and translate it if it's not in english
            original_text = doc_data.get('article') 
            translated_text = translator.translate(original_text) if detect(original_text) != "en" else original_text  
            #get district from the translated article
            district = next((word for word in translated_text.split() if word.lower() in lebanese_places), None)
            logger.info(f"Found new document: {doc.id}")
            logger.info(f"Original text: {original_text[:100]}...")
            logger.info(f"Translated text: {translated_text[:100]}...")
            logger.info(f"Detected district: {district}")
            new_articles.append({
                'id': doc.id,
                'article': original_text,
                'district': district
            })
        
        logger.info(f"Found {len(new_articles)} new articles to process")
        return new_articles
        
    except Exception as e:
        logger.error(f"Error in check_new_articles: {str(e)}")
        return []

async def process_articles():
    logger.info("Starting background article processing task")
    while True:
        try:
            new_articles = await check_new_articles()
            
            for article_data in new_articles:
                article = article_data['article']
                district = article_data['district']
                doc_id = article_data['id']
                
                if article:  # Skip if None or empty
                    logger.info(f"Processing article: {article[:50]}...")
                    result = classify_article(article)
                    
                    # Only process if classified as threatening
                    if result.get("threatening"):
                        logger.info("Article classified as threatening")
                        classification_data = result.get("category")
                        classification = (
                            classification_data
                            if isinstance(classification_data, list)
                            else [classification_data]
                        )
                        # Prepare document for classified_news collection
                        doc_data = {
                            "article": article,
                            "classification": classification,
                            "district": district,
                            "timestamp": datetime.now()
                        }
                        
                        # Save to classified_news collection
                        db.collection("classified_news").add(doc_data)
                        logger.info("Saved to classified_news collection")
                    
                    # Mark document as processed regardless of classification
                    db.collection('processed_articles').document(doc_id).set({
                        'processed_at': datetime.now(),
                        'was_threatening': result.get("threatening", False)
                    })

            
            # Wait for 2 seconds before next check
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error in process_articles: {str(e)}")
            await asyncio.sleep(1)  # Wait before retrying

@app.get("/status/")
async def status():
    try:
        processed_count = len(db.collection('processed_articles').get())
        return {
            "status": "running", 
            "processed_documents": processed_count,
            "background_task_running": True
        }
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return {"status": "error", "message": str(e)}

# Endpoint to check classified_news collection
@app.get("/check_classified_news/")
async def check_classified_news():
    try:
        docs = db.collection("classified_news").get()
        classified_articles = []
        for doc in docs:
            classified_articles.append(doc.to_dict())
        logger.info(f"Found {len(classified_articles)} classified articles")
        return {"status": "success", "classified_articles": classified_articles}
    except Exception as e:
        logger.error(f"Error checking classified news: {str(e)}")
        return {"status": "error", "message": str(e)}

