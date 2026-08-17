import easyocr
import torch
from PIL import Image
import open_clip
import numpy as np

reader = easyocr.Reader(['urdu','en'], gpu=False) 

DEBUG=True


def extractTextFromImage(image: str) -> list[str]:

    if not image:
        raise ValueError("Image path cannot be empty.")

    "Detail=0 make sure we dont get bounding box in output"

    result = reader.readtext(image, detail=0, gpu=False)
    if result is None:
        raise ValueError("OCR failed")
    
    return result


def getClipEmbedding(image: str, model: str = 'ViT-B-32') -> np.ndarray:

    try:


        # load the model
        model, _, preprocess = open_clip.create_model_and_transforms(model)
        # model is in training mode by default
        model.eval() 
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
    

    embedding = getClipEmbedding(image)

    if DEBUG==True:    
        print("\n--- CLIP Embedding ---")
        print(embedding[:5])  
        print(f"Embedding shape: {embedding.shape}")
    
    # Save embedding to a file for later use
    np.save('saveEmbedding.npy', embedding)