from fastapi import FastAPI, File, Form, HTTPException
from fastapi.responses import JSONResponse
import json
from typing import Optional
from PIL import Image
import io
from pathlib import Path
from datetime import datetime

from mlbox.services.peanuts import peanuts
from mlbox.services.peanuts.datatype import PeanutInputJson, PeanutProcessingRequest
from mlbox.settings import ROOT_DIR

app = FastAPI(
    title="MLBox Peanuts Service",
    description="Service for processing peanut images and generating analysis reports",
    version="1.0.0"
)

# Temporary directory for saving files during processing
TMP_DIR = ROOT_DIR / "tmp" / "peanuts" / "output"
TMP_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/peanuts/process_image")
async def process_image(
    image: bytes = File(...),
    json_data: str = Form(...)
):
    try:
        # Parse the JSON data
        json_body = json.loads(json_data)
        peanut_input_json = PeanutInputJson.from_json(json.dumps(json_body))
        
        # Convert bytes to PIL Image
        image_pil = Image.open(io.BytesIO(image))
        
        # Create processing request
        request = PeanutProcessingRequest(
            image=image_pil,
            alias=peanut_input_json.alias,
            key=peanut_input_json.key,
            response_method=peanut_input_json.response_method,
            response_endpoint=peanut_input_json.response_endpoint,
            image_filename=peanut_input_json.image_filename
        )
        
        # Process the request
        results = peanuts.process_requests([request])
        result = results[0]  # Since we're processing one image at a time
        
        if result.status == "error":
            raise HTTPException(status_code=400, detail=result.error_message)
            
        return JSONResponse(
            content={
                "status": "Success",
                "message": "Processing completed successfully.",
                "service_name": "peanuts",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "alias": request.alias,
                    "key": request.key,
                    "image_filename": request.image_filename,
                    "excel_file": result.excel_file
                }
            }
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 