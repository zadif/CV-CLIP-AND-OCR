from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import watchdog.events
import watchdog.observers
from fastapi.staticfiles import StaticFiles
from main import processAlreadyPresentImages, Handler,searchThroughImages
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")

observer = None

@app.on_event("startup")
async def startup_event():
    global observer
    path = "C:\\Users\\zadif\\Desktop\\VS CODE\\python\\CV\\CLIPXOCR\\images"
    
    processAlreadyPresentImages(path)
    
    handler = Handler()
    observer = watchdog.observers.Observer()
    observer.schedule(handler, path=path, recursive=True)
    observer.start()
    print("Server started. Observing the folder...")

@app.on_event("shutdown")
async def shutdown_event():
    global observer
    if observer:
        observer.stop()
        observer.join()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"results": []})

@app.post("/search", response_class=HTMLResponse)
async def do_search(request: Request, query: str = Form(...)):
    raw_results = searchThroughImages(query, topK=5)
    
    formattedResults = []
    for res in raw_results:
        formattedResults.append({
            "score": res.score,
            "path": os.path.basename(res.payload["image_path"]),
            "ocr_text": res.payload['ocr_text']
        })
        
    return templates.TemplateResponse(request, "index.html", {"request": request, "results": formattedResults, "query": query})




if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)