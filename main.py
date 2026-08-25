import easyocr
import torch
from PIL import Image
import open_clip
import numpy as np
import time
import watchdog.events
import watchdog.observers
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FilterSelector, Filter, FieldCondition, MatchValue, MatchText
import uuid
from PIL import Image
from pillow_heif import register_heif_opener
from config import loadConfig
import re
import difflib

config = loadConfig()

# solving the .avif issue
register_heif_opener()

reader = easyocr.Reader(config["ocr_languages"], gpu=False) 
openclipModel= config["clip_model"]
model, _, preprocess = open_clip.create_model_and_transforms(openclipModel, pretrained='laion2b_s34b_b88k')
model.eval() 

DEBUG=True

# Setting up the qdrant client
collectionName="images"
qdrant=QdrantClient(path="./qdrant")
if not (qdrant.collection_exists(collection_name=collectionName)):
    qdrant.create_collection(
        collection_name=collectionName,
        vectors_config=VectorParams(
            size=512,
            distance=Distance.COSINE
        )
    )

namespace=uuid.NAMESPACE_DNS

def extractTextFromImage(image: str) -> list[str]:
    try:
        if not image:
            raise ValueError("Image path cannot be empty.")

        # 1. Open with PIL
        img = Image.open(image)
        img = img.convert("RGB")

        img_np = np.array(img)

        "Detail=0 make sure we dont get bounding box in output"
        result = reader.readtext(img_np, detail=0)

        if result is None:
            raise ValueError("OCR failed")
        
        return result
    except Exception as e:
        print(f"Error occurred: {e}")
        raise


def getClipImageEmbedding(image: str) -> np.ndarray:

    try:

        # preprocess the image , convert to tensor , resize the image
        image = preprocess(Image.open(image)).unsqueeze(0)

        # if gpu is present then we will use it
        if torch.cuda.is_available():
            model.to("cuda")

            # it doesn't happen inplace
            image=image.to("cuda")


        # generate embeddings it returns a tensor of shape [1,512]
        image_features = model.encode_image(image)

        #normalize for cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # detach the tensor from the computation graph and move it to CPU, then convert to numpy array
        image_features = image_features.detach().cpu().numpy() 
       
    except Exception as e:
        print(f"Error occurred: {e}")
        raise



    return image_features


def processImage(image: str):
    try:
        time.sleep(0.05) 

        if DEBUG==True:    
            print(f"Processing: {image}")
        
        texts = extractTextFromImage(image)
        embedding = getClipImageEmbedding(image)

        config=loadConfig()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.flatten().tolist(),
            payload={
                "image_path": image,
                "ocr_text": texts,
                "watchFolder": config["path"]
            }
        )
        
        qdrant.upsert(collection_name=collectionName, points=[point])

        if DEBUG==True:  
            print(f"Added {image} to qdrant.")

    except Exception as e:
        print(f"Error occurred: {e}")
        raise

def splitTextQueries(text_queries: list[str]) -> list[str]:
    """
    Split multi-word quoted phrases into individual words.
    
    "The Doosra Prespective"  → ["the", "doosra", "prespective"]
    "vscode"                  → ["vscode"]
    "hello world"             → ["hello", "world"]
    """
    words = []
    for phrase in text_queries:
        for word in phrase.split():
            w = word.strip().lower()
            if w:
                words.append(w)
    return words

def fuzzyMatchOCR(
    query_words: list[str],
    ocr_words: list[str],
    threshold: float = 0.7
) -> float:
    """
    Compare each query WORD against each OCR WORD.
    Returns the best similarity ratio found (if >= threshold), else 0.0.

    Strategies (in priority order):
      1. Exact match       → 1.0
      2. Prefix match      → 0.75–1.0   ("per" starts "perspective")
      3. Substring match   → 0.65–0.9   ("spect" inside "perspective")
      4. Fuzzy/typo match  → 0.7–1.0    ("prespective" ↔ "perspective")
    """
    best_ratio = 0.0

    for q_word in query_words:
        q = q_word.lower().strip()
        if not q:
            continue

        for ocr_word in ocr_words:
            o = ocr_word.lower().strip()
            if not o:
                continue

            # 1. Exact
            if q == o:
                return 1.0

            # 2. Prefix: "per" starts "perspective"
            if o.startswith(q):
                score = 0.75 + (len(q) / len(o) * 0.25)
                if score > best_ratio:
                    best_ratio = score
                continue

            # 3. Substring: "spect" inside "perspective"
            if q in o:
                score = 0.65 + (len(q) / len(o) * 0.25)
                if score > best_ratio:
                    best_ratio = score
                continue

            # 4. Fuzzy: "prespective" ↔ "perspective" → 0.91
            ratio = difflib.SequenceMatcher(None, q, o).ratio()
            if ratio > best_ratio:
                best_ratio = ratio

    return best_ratio if best_ratio >= threshold else 0.0
    
def parseSearchQuery(query: str) -> tuple[str, list[str]]:

    # Extract everything inside double quotes
    text_queries = re.findall(r'"([^"]*)"', query)

    # Remove all quoted segments (including the quotes) from original query
    clip_query = re.sub(r'"[^"]*"', '', query)

    # Clean up leftover whitespace / double spaces
    clip_query = re.sub(r'\s+', ' ', clip_query).strip()

    return clip_query, text_queries

class SearchResult:
    def __init__(self, score: float, payload: dict, source: str = "vector"):
        self.score = score
        self.payload = payload
        self.source = source   # "vector", "ocr", "both"

def searchThroughImages(query: str, topK: int = 3, watchFolder: str = None,fuzzy_enabled: bool = False  ) -> list[SearchResult]:

    try:

        clip_query, text_queries = parseSearchQuery(query)

        # Build the folder filter , we use it to match the folder from the config and the folder we have added in the point in qdrant
        folder_condition = None
        if watchFolder:
            folder_condition = FieldCondition(
                key="watchFolder",
                match=MatchValue(value=watchFolder)
            )

        # CLIP search
        vector_results = None
        query_vector = None

        if clip_query:
            tokenizer = open_clip.get_tokenizer(openclipModel)
            text = tokenizer(clip_query)

            if torch.cuda.is_available():
                model.to("cuda")
                text = text.to("cuda")

            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.detach().cpu().numpy()
            query_vector = text_features.flatten()

            vector_query_filter = None
            if folder_condition:
                vector_query_filter = Filter(must=[folder_condition])

            vector_results = qdrant.query_points(
                collection_name=collectionName,
                query=query_vector.tolist(),
                limit=topK * 2,
                query_filter=vector_query_filter
            )

        #  OCR Search
        ocr_exact_ids = set()
        ocr_results = []

        if text_queries:
            # Split all quoted phrases into individual words
            query_words = splitTextQueries(text_queries)

            # ── Tier 1: Qdrant PREFIX match (per-word, OR logic) ──
            # Each word is a separate condition in "should" (OR).
            # So a typo in one word doesn't kill the entire search.
            text_conditions = [
                FieldCondition(key="ocr_text", match=MatchText(text=w))
                for w in query_words
            ]

            ocr_filter_must = []
            if folder_condition:
                ocr_filter_must.append(folder_condition)

            try:
                exact_scroll, _ = qdrant.scroll(
                    collection_name=collectionName,
                    scroll_filter=Filter(
                        must=ocr_filter_must,
                        should=text_conditions   # OR: at least ONE word matches
                    ),
                    limit=topK * 3,
                    with_vectors=True
                )
                for point in exact_scroll:
                    ocr_exact_ids.add(point.id)
                    ocr_results.append(point)
            except Exception as e:
                print(f"OCR exact search failed: {e}")

            # ── Tier 2: Fuzzy match (only if Tolerance is ON) ──
            if fuzzy_enabled:
                try:
                    all_scroll_filter = None
                    if folder_condition:
                        all_scroll_filter = Filter(must=[folder_condition])

                    all_points, _ = qdrant.scroll(
                        collection_name=collectionName,
                        scroll_filter=all_scroll_filter,
                        limit=10000,
                        with_vectors=True
                    )

                    for point in all_points:
                        if point.id in ocr_exact_ids:
                            continue  # already found by Tier 1

                        ocr_words = point.payload.get("ocr_text", [])
                        if not ocr_words:
                            continue

                        # Word-to-word fuzzy comparison
                        fuzzy_score = fuzzyMatchOCR(query_words, ocr_words, threshold=0.7)

                        if fuzzy_score > 0.0:
                            point._fuzzy_score = fuzzy_score
                            ocr_results.append(point)

                except Exception as e:
                    print(f"OCR fuzzy search failed: {e}")

        # merging both results
        merged = {}

        # Add vector search results
        if vector_results:
            for point in vector_results.points:
                merged[point.id] = {
                    "score": point.score,
                    "payload": point.payload,
                    "source": "vector"
                }

        # Add OCR results
        for point in ocr_results:
            if point.id in merged:
                #  Found in BOTH → boost score, it is given more priority
                boosted = min(1.0, merged[point.id]["score"] + 0.05)
                merged[point.id] = {
                    "score": boosted,
                    "payload": point.payload,
                    "source": "both"
                }
            else:
                # Found only via OCR
                # Compute cosine similarity if we have a query vector
                if query_vector is not None and point.vector:
                    stored_vec = np.array(point.vector)
                    cosine_sim = float(np.dot(query_vector, stored_vec))
                    score = max(0.0, cosine_sim)
                else:
                    # No query
                    score = 1.0

                merged[point.id] = {
                    "score": score,
                    "payload": point.payload,
                    "source": "ocr"
                }

        sorted_results = sorted(
            merged.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [
            SearchResult(
                score=r["score"],
                payload=r["payload"],
                source=r["source"]
            )
            for r in sorted_results[:topK]
        ]

    except Exception as e:
        print(f"Error occurred: {e}")
        raise

class Handler(watchdog.events.PatternMatchingEventHandler):

    def __init__(self):
        super().__init__(
            patterns=["*.jpg", "*.png", "*.jpeg","*.webp","*.avif"],
            ignore_directories=True,
            case_sensitive=False
        )

    def on_created(self, event):
        try:
            processImage(event.src_path)

        except Exception as e:
            print(f"Error occurred: {e}")
            raise


    def on_deleted(self, event):
        global database
        try:
            qdrant.delete(
                collection_name=collectionName,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="image_path",
                                match=MatchValue(value=event.src_path)
                            )
                        ]
                    )
                )
            )
            if DEBUG == True:
                print(f"Removed {event.src_path} from qdrant.")

        except Exception as e:
            print(f"Error occurred: {e}")
            raise

def isAlreadyPresentInQdrant(image: str) -> bool:

    results, _ = qdrant.scroll(
        collection_name=collectionName,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="image_path",
                    match=MatchValue(value=image)
                )
            ]
        ),
        limit=1
    )
    return len(results) > 0

def processAlreadyPresentImages(folderPath:str):
    # processing already present images if any
    
    files=os.listdir(folderPath)
    imageFiles = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg',".webp",".avif"))]

    for file in imageFiles:
        name=os.path.join(folderPath, file)

        if isAlreadyPresentInQdrant(name):
            continue
        processImage(name)


