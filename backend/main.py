from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = FastAPI()

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = None
model = None
generator = None

class EmailRequest(BaseModel):
    prompt: str

@app.on_event("startup")
def load_model():
    global tokenizer, model, generator
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.post("/generate_email")
def generate_email(request: EmailRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    prompt = request.prompt
    try:
        outputs = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
        return {"email": outputs[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Testing model loading with a sample prompt...")
    test_prompt = "Write a professional follow-up email to a client about an overdue invoice."
    load_model()
    outputs = generator(test_prompt, max_new_tokens=128, do_sample=True, temperature=0.7)
    print("Sample output:\n", outputs[0]["generated_text"])
    uvicorn.run(app, host="0.0.0.0", port=8000)
