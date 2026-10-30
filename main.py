from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
import json

app = FastAPI(title="Terminal-Bench Auto-Debugger")

# Allow frontend requests during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Nebius Client (OpenAI compatible)
client = AsyncOpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.getenv("NEBIUS_API_KEY")
)

class DebugRequest(BaseModel):
    code_snippet: str
    error_message: str = "No error message provided. Analyze for logical or syntax flaws."

@app.post("/api/debug")
async def debug_code(request: DebugRequest):
    if not request.code_snippet.strip():
        raise HTTPException(status_code=400, detail="Code snippet cannot be empty.")

    system_prompt = """
    You are an expert backend software developer debugging code. 
    Analyze the provided code and error message. 
    Return ONLY a valid JSON object with exactly two keys:
    1. "diagnosis": A brief, technical explanation of what went wrong.
    2. "fixed_code": The corrected version of the code.
    Do not include any markdown formatting, backticks, or extra text outside the JSON object.
    """

    user_prompt = f"Code:\n{request.code_snippet}\n\nError Log:\n{request.error_message}"

    try:
        response = await client.chat.completions.create(
            model="nvidia/nemotron-4-340b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, 
        )
        
        result_text = response.choices[0].message.content.strip()
        parsed_result = json.loads(result_text)
        
        return parsed_result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model failed to return structured JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))