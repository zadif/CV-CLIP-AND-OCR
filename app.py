from tkinter import filedialog
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import watchdog.events
import watchdog.observers
from fastapi.staticfiles import StaticFiles
from main import processAlreadyPresentImages, Handler,searchThroughImages
import os
from config import loadConfig,saveConfig
import tkinter as tk
from fastapi.responses import FileResponse

config = loadConfig()
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")

observer = None

@app.on_event("startup")
async def startup_event():
    global observer
    path = config["path"]
    
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
    return templates.TemplateResponse(request, "index.html", {"results": [], "config": config})

@app.post("/search", response_class=HTMLResponse)
async def do_search(request: Request, query: str = Form(...)):
    config=loadConfig()
    raw_results = searchThroughImages(query, topK=config["topK"], watchFolder=config["path"])
    
    formattedResults = []
    for res in raw_results:
        formattedResults.append({
            "score": res.score,
            "path": res.payload["image_path"],
            "ocr_text": res.payload['ocr_text']
        })
        
    return templates.TemplateResponse(request, "index.html", {"request": request, "results": formattedResults, "query": query, "config": config})


@app.get("/api/settings")
async def get_settings():
    return config

@app.post("/api/settings")
async def updateSettings(request: Request, watch_folder: str = Form(...), clip_model: str = Form(...), ocr_languages: str = Form(...), recursive_watch: bool = Form(False), topK: int = Form(...)):
    global config,observer
    oldFolder=config["path"]

    config["path"] = watch_folder
    config["clip_model"] = clip_model
    config["ocr_languages"] = [lang.strip() for lang in ocr_languages.split(",")]
    config["recursive_watch"] = recursive_watch
    config["topK"] = topK

    saveConfig(config)
    print(oldFolder)
    print(watch_folder)
    if oldFolder != watch_folder:
        print(f"Folder changed from {oldFolder} to {watch_folder}. Restarting watcher...")
        
        # Stop the old watcher
        if observer:
            observer.stop()
            observer.join()
            
        # Start a new watcher on the new folder
        handler = Handler()
        observer = watchdog.observers.Observer()
        observer.schedule(handler, path=watch_folder, recursive=recursive_watch)
        observer.start()
        print(f"Now watching: {watch_folder}")


    
    return {"status": "success", "message": "Settings updated successfully!"}

@app.get("/api/pick-folder")
async def pick_folder():
    # Open native Windows folder picker
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Select Folder to Watch")
    root.destroy()
    
    if folder_selected:
        global config
        config["path"] = folder_selected
        saveConfig(config)
        return {"status": "success", "folder": folder_selected}
    return {"status": "cancelled"}


@app.get("/view-image")
async def view_image(file_path: str):
    return FileResponse(file_path)



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)