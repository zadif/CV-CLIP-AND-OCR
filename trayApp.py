import threading
import uvicorn
import pystray
from PIL import Image, ImageDraw
from app import app 
import webbrowser

def runServer():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")


def createIcon():
    img = Image.new('RGB', (64, 64), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((10, 10, 54, 54), fill=(255, 130, 0))
    return img

def openBrowser(icon, item):
    webbrowser.open("http://127.0.0.1:8000")

def quitApp(icon, item):
    icon.stop()

icon = pystray.Icon(
    "CLIP_Search",
    createIcon(),
    "Image Search",
    menu=pystray.Menu(
        pystray.MenuItem("Open Search", openBrowser, default=True), # Double-click opens browser
        pystray.MenuItem("Quit", quitApp)
    )
)

if __name__ == "__main__":

    # Start the Uvicorn server in a daemon thread (daemon=True means it dies when main script dies)
    server_thread = threading.Thread(target=runServer, daemon=True)
    server_thread.start()

    print("Starting background server... Look for the icon in your system tray!")
    
    # Run the tray icon (this blocks the main thread and keeps the script alive)
    icon.run()