import easyocr
import torch
from PIL import Image
import open_clip
import numpy as np

reader = easyocr.Reader(['en'], gpu=False) 
openclipModel= 'ViT-B-32'
model, _, preprocess = open_clip.create_model_and_transforms(openclipModel, pretrained='openai')
model.eval() 

DEBUG=True


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


def processScreenshot(image: str):

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
    
    # Save embedding to a file for later use
    np.save('saveEmbedding.npy', embedding)


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


if __name__ == "__main__":
    # 1. Define a list of test images (put 3-4 real images in a folder)
    test_images = ["C:\\Users\\zadif\\Desktop\\VS CODE\\python\\CV\\CLIPXOCR\\images\\1.jpg", "C:\\Users\\zadif\\Desktop\\VS CODE\\python\\CV\\CLIPXOCR\\images\\2.jpg", "C:\\Users\\zadif\\Desktop\\VS CODE\\python\\CV\\CLIPXOCR\\images\\3.jpg"]  # Replace with your actual image paths
    
    # 2. Build the temporary database
    database = []
    for img_path in test_images:
        # Call your functions to get the data
        ocr_text = extractTextFromImage(img_path)
        clip_emb = getClipImageEmbedding(img_path)
        
        # Append a dictionary to the database list
        database.append({
            "image_path": img_path,
            "ocr_text": ocr_text,
            "clipEmbedding": clip_emb
        })
        print(f"Added {img_path} to database.")

    print("\n--- Starting Search ---")
    # 3. Test the search function
    query = "a picture of a man"  # Change this to match one of your test images
    results = searchThroughImages(query, database, topK=3)
    
    # 4. Print the results
    for score, item in results:
        print(f"Score: {score:.4f} | Path: {item['image_path']}")