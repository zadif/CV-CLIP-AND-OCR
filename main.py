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
from qdrant_client.models import Distance, VectorParams, PointStruct, FilterSelector, Filter, FieldCondition, MatchValue
import uuid


reader = easyocr.Reader(['en'], gpu=False) 
openclipModel= 'ViT-B-32'
model, _, preprocess = open_clip.create_model_and_transforms(openclipModel, pretrained='openai')
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

        "Detail=0 make sure we dont get bounding box in output"

        result = reader.readtext(image, detail=0)
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

        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.flatten().tolist(),
            payload={
                "image_path": image,
                "ocr_text": texts
            }
        )
        
        qdrant.upsert(collection_name=collectionName, points=[point])

        if DEBUG==True:  
            print(f"Added {image} to qdrant.")

    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    



def searchThroughImages(query: str, topK: int = 3) -> list[dict]:

    try:

        # Load tokenizer
        tokenizer = open_clip.get_tokenizer(openclipModel)
        
        # Tokenize the query 
        text = tokenizer(query)

        if torch.cuda.is_available():
                    model.to("cuda")
        
                    # it doesn't happen inplace
                    text=text.to("cuda")

        # Generate text embedding
        text_features = model.encode_text(text)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.detach().cpu().numpy() 
        
        results = qdrant.search(
            collection_name=collectionName,
            query_vector=text_features.flatten().tolist(),
            limit=topK
        )

        # 7. Return the topK results.
        return results

    except Exception as e:
        print(f"Error occurred: {e}")
        raise


class Handler(watchdog.events.PatternMatchingEventHandler):

    def __init__(self):
        super().__init__(
            patterns=["*.jpg", "*.png", "*.jpeg"],
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

if __name__ == "__main__":

    path = "C:\\Users\\zadif\\Desktop\\VS CODE\\python\\CV\\CLIPXOCR\\images"

    # processing already present images if any
    files=os.listdir(path)
    imageFiles = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for file in imageFiles:
        name=os.path.join(path, file)

        if isAlreadyPresentInQdrant(name):
            continue
        processImage(name)

    handler = Handler()
    observer = watchdog.observers.Observer()
    observer.schedule(handler, path=path, recursive=True)
    observer.start()
    print("Observing the folder")

 


    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
   