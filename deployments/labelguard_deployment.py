from loguru import logger
import json
import sys
import traceback
import time
import asyncio
from io import BytesIO
from typing import List
from pathlib import Path
import ray
from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate
from starlette.responses import Response
from urllib.parse import unquote

from PIL import Image as PILImage
from ray import serve
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from mlbox.services.LabelGuard.datatypes import LabelInput
from mlbox.services.LabelGuard import labelguard
from mlbox.settings import ROOT_DIR
from mlbox.utils.logger import get_logger, get_artifact_service
import uuid

app_logger = get_logger(ROOT_DIR)
artifact_service = get_artifact_service(ROOT_DIR)


@serve.deployment
class LabelGuard:
    SERVICE_NAME = "labelguard"
    
    def __init__(self):
        # Temporary directory for saving files during processing
        self.tmp_dir = ROOT_DIR / "assets" / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        
        app_logger.info(self.SERVICE_NAME, f"LabelGuard deployment initialized. LOG_LEVEL={app_logger.level}")
    
    @staticmethod
    def create_response(status, message, data=None, error_trace=None) -> dict:
        response_data = {
            "status": status,
            "message": message,
        }
        
        if data:
            response_data["data"] = data
        
        if error_trace:
            response_data["stack"] = error_trace
        
        return response_data
    
    async def __call__(self, request: Request):
        # Route based on path
        path = request.url.path
        app_logger.info(self.SERVICE_NAME, f"Received request: {request.method} {path}")
        
        # Strip /labelguard prefix if present (Ray Serve route_prefix)
        if path.startswith("/labelguard/"):
            path = path[len("/labelguard"):]  # Remove /labelguard but keep the leading /
        
        # Handle /analyze endpoint
        if path.endswith("/analyze") or path.endswith("/analyze/"):
            app_logger.info(self.SERVICE_NAME, f"Routing to analyze_endpoint for: {path}")
            return await self.analyze_endpoint(request)
        
        # Handle static file serving for /artifacts/ paths
        if path.startswith("/artifacts/"):
            app_logger.info(self.SERVICE_NAME, f"Routing to serve_static_file for: {path}")
            return await self.serve_static_file(request)
        
        # Default: legacy batch handler
        accept_header = request.headers.get("accept", "")
        result = await self.batch_handler(request)
        
        # If single result and HTML report exists, return HTML
        if len(result) == 1 and isinstance(result[0], dict):
            if result[0].get("status") == "success" and "html_report" in result[0].get("data", {}):
                if "text/html" in accept_header or request.query_params.get("format") == "html":
                    return HTMLResponse(content=result[0]["data"]["html_report"])
        
        return result[0] if len(result) == 1 else result
    
    async def analyze_endpoint(self, request: Request):
        """Handle POST /labelguard/analyze endpoint"""
        # Handle CORS preflight
        if request.method == "OPTIONS":
            return JSONResponse(
                {},
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                }
            )
        
        if request.method != "POST":
            return JSONResponse(
                {"status": "error", "message": "Method not allowed"},
                status_code=405,
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        try:
            # Parse request body
            body = await request.json()
            
            # Extract parameters
            image_path = body.get("image_path")
            blocks = body.get("blocks", [])
            kmat = body.get("kmat", "UNKNOWN")
            version = body.get("version", "v1.0")
            etalon = body.get("etalon")
            
            app_logger.info(self.SERVICE_NAME, f"Analyze request: kmat={kmat}, version={version}, blocks={len(blocks)}")
            
            # Check if this is first call (empty blocks) or subsequent call
            is_first_call = len(blocks) == 0
            
            # Prepare request JSON for analyze function
            request_json = {
                "image_path": image_path,
                "blocks": blocks,
                "kmat": kmat,
                "version": version
            }
            if etalon:
                request_json["etalon"] = etalon
            
            # Call analyze function
            result = labelguard.analyze(request_json)
            
            # Determine original filename
            original_filename = result.original_filename or Path(image_path).stem if image_path else "unknown"
            
            # Artifacts directory
            artifacts_dir = ROOT_DIR / "artifacts" / "labelguard"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image and JSON to artifacts
            image_file = artifacts_dir / f"{original_filename}.jpg"
            json_file = artifacts_dir / f"{original_filename}.json"
            
            # Load and save image
            if image_path:
                image_full_path = ROOT_DIR / image_path.lstrip('/')
                if image_full_path.exists():
                    from shutil import copyfile
                    copyfile(str(image_full_path), str(image_file))
                    app_logger.info(self.SERVICE_NAME, f"Saved image to {image_file}")
            
            # Serialize result to JSON
            result_json = result.to_json()
            
            # Save JSON
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "image_path": f"/artifacts/labelguard/{original_filename}.jpg",
                    "labelProcessingResult": result_json
                }, f, ensure_ascii=False, indent=2)
            app_logger.info(self.SERVICE_NAME, f"Saved JSON to {json_file}")
            
            # Return response based on call type
            cors_headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
            
            if is_first_call:
                # First call: return file paths
                return JSONResponse({
                    "status": "success",
                    "image_path": f"/artifacts/labelguard/{original_filename}.jpg",
                    "data_endpoint": f"/artifacts/labelguard/{original_filename}.json",
                    "original_filename": original_filename
                }, headers=cors_headers)
            else:
                # Subsequent call: return LabelProcessingResult JSON
                return JSONResponse({
                    "status": "success",
                    "labelProcessingResult": result_json
                }, headers=cors_headers)
                
        except Exception as e:
            error_trace = traceback.format_exc()
            app_logger.error(self.SERVICE_NAME, f"Analyze endpoint error: {str(e)}\n{error_trace}")
            return JSONResponse(
                {
                    "status": "error",
                    "message": str(e),
                    "stack": error_trace
                },
                status_code=500,
                headers={"Access-Control-Allow-Origin": "*"}
            )
    
    async def serve_static_file(self, request: Request):
        """Serve static files from /artifacts/ directory"""
        
        path = request.url.path
        app_logger.info(self.SERVICE_NAME, f"serve_static_file called with path: {path}")
        
        # Strip /labelguard prefix if present (Ray Serve route_prefix)
        if path.startswith("/labelguard/"):
            path = path[len("/labelguard"):]  # Remove /labelguard but keep the leading /
        
        # Remove leading /artifacts/ to get relative path
        if path.startswith("/artifacts/"):
            relative_path = path[len("/artifacts/"):]
        else:
            relative_path = path.lstrip("/")
        
        # URL decode the path to handle special characters (Cyrillic, etc.)
        relative_path = unquote(relative_path)
        
        # Construct full file path
        file_path = ROOT_DIR / "artifacts" / relative_path
        
        app_logger.debug(self.SERVICE_NAME, f"Serving static file: {file_path} (exists: {file_path.exists()})")
        
        # Security check: ensure path is within artifacts directory
        try:
            file_path.resolve().relative_to(ROOT_DIR / "artifacts")
        except ValueError:
            return JSONResponse(
                {"status": "error", "message": "Invalid path"},
                status_code=403
            )
        
        if not file_path.exists():
            app_logger.warning(self.SERVICE_NAME, f"File not found: {file_path}")
            return JSONResponse(
                {"status": "error", "message": f"File not found: {relative_path}"},
                status_code=404
            )
        
        # Determine content type
        if file_path.suffix.lower() in ['.jpg', '.jpeg']:
            media_type = "image/jpeg"
        elif file_path.suffix.lower() == '.png':
            media_type = "image/png"
        elif file_path.suffix.lower() == '.gif':
            media_type = "image/gif"
        elif file_path.suffix.lower() == '.json':
            media_type = "application/json"
        else:
            media_type = "application/octet-stream"
        
        # Read file content
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Return response with CORS headers
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Access-Control-Allow-Origin": "*",  # Allow all origins for development
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    
    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def batch_handler(self, requests: List[Request]) -> List[dict]:
        app_logger.info(self.SERVICE_NAME, f"Batch processing started | batch_size={len(requests)}")
        
        label_inputs: List[LabelInput] = []
        responses = [None] * len(requests)
        request_ids = [None] * len(requests)
        start_times = [None] * len(requests)
        requests_mapping = {}
        
        for idx, request in enumerate(requests):
            # Log each request BEFORE processing
            client_ip = request.client.host if request.client else "unknown"
            app_logger.info(self.SERVICE_NAME, f"Processing request {idx+1}/{len(requests)} | client_ip={client_ip}")
            
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            try:
                form = await request.form()
                
                image_file = form["image"]
                if image_file is None:
                    raise Exception(f"IMAGE MISSING in form | request_id={request_id}")
                
                image_data = await image_file.read()
                
                # Save input image
                image = PILImage.open(BytesIO(image_data))
                
                json_body = json.loads(form["json"])
                
                # Extract kmat and version from JSON
                kmat = json_body.get("kmat", "UNKNOWN")
                version = json_body.get("version", "v1.0")
                
                # Create temporary image file with proper naming for etalon lookup
                # Use the filename from form, or construct from kmat_version
                temp_image_path = self.tmp_dir / f"{request_id}_{image_file.filename}"
                image.save(temp_image_path)
                
                # Check if etalon file was provided in the request, otherwise look for it
                etalon_file = form.get("etalon")
                if etalon_file:
                    # Save etalon file alongside the image with same naming pattern
                    # temp_image_path is like: /tmp/uuid_filename.jpg
                    # etalon should be: /tmp/uuid_filename_etalon.json
                    etalon_path = temp_image_path.parent / f"{temp_image_path.stem}_etalon.json"
                    etalon_data = await etalon_file.read()
                    with open(etalon_path, 'wb') as f:
                        f.write(etalon_data)
                    app_logger.info(self.SERVICE_NAME, f"Saved etalon file: {etalon_path}")
                
                # Log request to app.log
                app_logger.info(self.SERVICE_NAME, f"Request received | kmat={kmat} | version={version}")
                
                # Save input JSON as artifact
                artifact_service.save_artifact(self.SERVICE_NAME, f"input_{request_id}.json", json_body)
                
                request_ids[idx] = request_id
                start_times[idx] = start_time
                
                # Save input image as artifact
                artifact_service.save_artifact(self.SERVICE_NAME, f"input_{request_id}_{image_file.filename}", image)
                
                # Create LabelInput object
                label_input = LabelInput(
                    kmat=kmat,
                    version=version,
                    label_image=image,
                    label_image_path=temp_image_path
                )
                
                label_inputs.append(label_input)
                requests_mapping[len(label_inputs) - 1] = idx
                
            except JSONSchemaValidationError as e:
                error_msg = f"JSON schema validation error: {str(e)}"
                responses[idx] = self.create_response(
                    status="error", message=error_msg
                )
                app_logger.error(self.SERVICE_NAME, f"request_id={request_id} | error={error_msg}")
            except Exception as e:
                responses[idx] = self.create_response(status="error", message=str(e))
                app_logger.error(self.SERVICE_NAME, f"request_id={request_id} | error={str(e)}")
        
        if label_inputs:
            try:
                # Process labels
                app_logger.info(self.SERVICE_NAME, f"Calling labelguard.process_labels with {len(label_inputs)} labels")
                processing_results = labelguard.process_labels(label_inputs)
                app_logger.info(self.SERVICE_NAME, f"Processing completed | results_count={len(processing_results)}")
                
                for idx, processing_result in enumerate(processing_results):
                    original_idx = requests_mapping[idx]
                    request_id = request_ids[original_idx]
                    start_time = start_times[original_idx]
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time if start_time else 0
                    
                    # Count errors/warnings from rule check results
                    total_errors = sum(1 for r in processing_result.rule_check_results if not r.passed)
                    total_checks = len(processing_result.rule_check_results)
                    
                    # Read the generated HTML report from artifacts
                    html_report = None
                    html_filename = f"{request_id}_{processing_result.original_filename}_interactive_viewer.html"
                    html_path = artifact_service.get_service_dir(self.SERVICE_NAME) / html_filename
                    if html_path.exists():
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_report = f.read()
                        app_logger.info(self.SERVICE_NAME, f"Read HTML report from {html_path}")
                    
                    # Create response data
                    response_data = {
                        "processing_time_seconds": processing_time,
                        "status": "success" if processing_result.success else "error",
                        "kmat": processing_result.kmat,
                        "version": processing_result.version,
                        "total_checks": total_checks,
                        "errors_count": total_errors,
                        "text_blocks_count": len(processing_result.text_blocks),
                        "html_report": html_report  # Include HTML report read from artifacts
                    }
                    
                    # Save response data as artifact
                    artifact_service.save_artifact(self.SERVICE_NAME, f"response_{request_id}.json", response_data)
                    
                    # Log response to app.log
                    status = response_data.get("status", "unknown")
                    app_logger.info(
                        self.SERVICE_NAME,
                        f"Response sent | request_id={request_id} | status={status} | "
                        f"errors={total_errors}/{total_checks} | time={processing_time:.2f}s"
                    )
                    
                    responses[original_idx] = self.create_response(
                        status="success",
                        message=f"Label validation completed: {total_errors} errors found in {total_checks} checks",
                        data=response_data
                    )
                
            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                error_trace = traceback.format_exc()
                app_logger.error(self.SERVICE_NAME, f"error={str(e)}\n{error_trace}")
                
                # Proper error fan-out: fill all pending responses on failure
                for idx, request_id in enumerate(request_ids):
                    if request_id and responses[idx] is None:
                        app_logger.error(self.SERVICE_NAME, f"Request failed | request_id={request_id} | error={str(e)}")
                        responses[idx] = self.create_response(
                            status="error", message=error_msg, error_trace=error_trace
                        )
        
        app_logger.info(self.SERVICE_NAME, f"Batch processing completed | batch_size={len(requests)}")
        return responses


# LabelGuard deployment binding
LabelGuardDeployment = LabelGuard.bind()

