import easyocr
import torch
from PIL import Image
import open_clip
import numpy as np
import time
import watchdog.events
import watchdog.observers
import os


reader = easyocr.Reader(['en'], gpu=False) 
openclipModel= 'ViT-B-32'
model, _, preprocess = open_clip.create_model_and_transforms(openclipModel, pretrained='openai')
model.eval() 

DEBUG=False
database = []


def extractTextFromImage(image: str) -> list[str]:

    if not image:
        raise ValueError("Image path cannot be empty.")

    "Detail=0 make sure we dont get bounding box in output"

    result = reader.readtext(image, detail=0)
    if result is None:
        raise ValueError("OCR failed")
    
    return result


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
        time.sleep(0.5) 

        if DEBUG==True:    
            print(f"Processing: {image}")
        
        texts = extractTextFromImage(image)

        if DEBUG==True:    
            print("\n--- OCR Text Detected ---")
            for i, text in enumerate(texts, 1):
                print(f"{i}. {text}")
        

        embedding = getClipImageEmbedding(image)

        if DEBUG==True:    
            print("\n--- CLIP Embedding ---")
            print(embedding[:5])  
            print(f"Embedding shape: {embedding.shape}")

        database.append({
            "image_path": image,
            "ocr_text": texts,
            "clipEmbedding": embedding
        })
        print(f"Added {image} to database. Total: {len(database)}")

    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    



def searchThroughImages(query: str, database: list[dict], topK: int = 3) -> list[dict]:

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
        
        # Loop through database. For each item, and calculate the similarity  
      
        scores=[]
        for item in database:
            score=np.dot(text_features[0], item["clipEmbedding"][0])
            scores.append((score,item))

        # Sort the results from highest to lowest score.
        scores.sort(key=lambda x: x[0], reverse=True)

        # 7. Return the topK results.
        return scores[:topK]

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
            database = [item for item in database if item["image_path"] != event.src_path]
            print(f"Removed {event.src_path} from database. Total images: {len(database)}")
        except Exception as e:
                        print(f"Error occurred: {e}")
                        raise


  



if __name__ == "__main__":

    path = "C:\\Users\\zadif\\Desktop\\VS CODE\\python\\CV\\CLIPXOCR\\images"

    # processing already present images
    files=os.listdir(path)
    imageFiles = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for file in imageFiles:
        name=os.path.join(path, file)
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
   