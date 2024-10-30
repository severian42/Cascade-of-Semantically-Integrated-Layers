from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from SemanticCascadeProcessing import SemanticCascadeProcessor, SCPConfig
from .middleware import verify_api_key
import time

app = FastAPI(title="SCP API")
app.middleware("http")(verify_api_key)
scp = SemanticCascadeProcessor(SCPConfig())

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 2048
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    stop: Optional[List[str]] = None

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # Extract the last user message
        last_message = request.messages[-1]
        if last_message.role != "user":
            raise HTTPException(status_code=400, message="Last message must be from user")
            
        # Process through SCP
        result = scp.process_interaction(last_message.content)
        
        # Format response like OpenAI
        response = ChatCompletionResponse(
            id=f"scp-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=result['final_response']
                    ),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": len(last_message.content.split()),
                "completion_tokens": len(result['final_response'].split()),
                "total_tokens": len(last_message.content.split()) + 
                              len(result['final_response'].split())
            }
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))