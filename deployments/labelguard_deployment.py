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

from PIL import Image as PILImage
from ray import serve
from starlette.requests import Request
from starlette.responses import HTMLResponse
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
        # Check if HTML response is requested
        accept_header = request.headers.get("accept", "")
        result = await self.batch_handler(request)
        
        # If single result and HTML report exists, return HTML
        if len(result) == 1 and isinstance(result[0], dict):
            if result[0].get("status") == "success" and "html_report" in result[0].get("data", {}):
                if "text/html" in accept_header or request.query_params.get("format") == "html":
                    return HTMLResponse(content=result[0]["data"]["html_report"])
        
        return result[0] if len(result) == 1 else result
    
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

