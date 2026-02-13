from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

def run():
    # You’ll parse --port, --data-dir, --log-dir later
    uvicorn.run("app.main:app", host="127.0.0.1", port=0)

if __name__ == "__main__":
    run()
