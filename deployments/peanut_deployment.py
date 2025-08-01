from loguru import logger

logger.info(">>> MODULE LOADED1")

import json
import sys
import traceback
import time
from io import BytesIO
from typing import List
from pathlib import Path
import ray
from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate


from PIL import Image as PILImage
from ray import serve
from starlette.requests import Request
from mlbox.services.peanuts.datatype import PeanutInputJson
from mlbox.services.peanuts import peanuts
from mlbox.settings import ROOT_DIR
from mlbox.utils.logger import get_logger, get_artifact_service
import uuid

# Initialize logger and artifact service
app_logger = get_logger(ROOT_DIR)
artifact_service = get_artifact_service(ROOT_DIR)

@serve.deployment
class Peanuts:
    SERVICE_NAME = "peanuts"
    def __init__(self):
        # Temporary directory for saving files during processing
        self.tmp_dir = ROOT_DIR / "assets" / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Load the JSON schema for peanut requests
        request_json_schema_file = (
            ROOT_DIR / "assets" / "json-schemas" / "peanut-input.json"
        )
        
        with open(request_json_schema_file, "r", encoding="utf-8") as schema_file:
            self.peanut_request_json_schema = json.load(schema_file)


    @staticmethod
    def create_response(status, message, error_trace=None) -> dict:
        response_data = {
            "status": status,
            "message": message + (f"\n{error_trace}" if error_trace else message),
        }

        return response_data

    async def __call__(self, request: Request):
        return await self.batch_handler(request)

    # @ray.remote
    def peanuts_process_requests(
        self, peanut_requests
    ) -> List[peanuts.PeanutProcessingResult]:
        return peanuts.process_requests(peanut_requests)

    @serve.batch(max_batch_size=5, batch_wait_timeout_s=0)
    async def batch_handler(self, requests: List[Request]) -> List[dict]:
        app_logger.info(self.SERVICE_NAME, f"Batch processing started | batch_size={len(requests)}")
        
        peanut_requests: List[peanuts.PeanutProcessingRequest] = []
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
                
                # Create temporary image file for logging
                temp_image_path = self.tmp_dir / f"input_{idx}_{image_file.filename}"
                image.save(temp_image_path)

                json_body = json.loads(form["json"])
                validate(instance=json_body, schema=self.peanut_request_json_schema)
                peanut_input_json = PeanutInputJson.from_json(json.dumps(json_body)) # pylint: disable=no-member

                # Log request to app.log
                app_logger.info(self.SERVICE_NAME, f"Request received | {peanut_input_json.dict()} ")
                
                # Save peanut_input_json as artifact
                artifact_service.save_artifact("peanuts", f"input_{request_id}.json", peanut_input_json.__dict__)
                
                
                request_ids[idx] = request_id
                start_times[idx] = start_time
                
                # Save input image as artifact
                artifact_service.save_artifact(self.SERVICE_NAME, f"input_{request_id}_{image_file.filename}", image)

                peanut_requests.append(
                    peanuts.PeanutProcessingRequest(
                        image=image,
                        alias=peanut_input_json.alias,
                        key=peanut_input_json.key,
                        image_filename=image_file.filename,
                        response_method=peanut_input_json.response_method,
                        response_endpoint=peanut_input_json.response_endpoint,
                    )
                )

                requests_mapping[len(peanut_requests) - 1] = idx

                responses[idx] = self.create_response(
                    status="received", message="Your request is being processed"
                )

            except JSONSchemaValidationError as e:
                error_msg = f"JSON schema validation error: {str(e)}"
                responses[idx] = self.create_response(
                    status="error", message=error_msg
                )
                app_logger.error(self.SERVICE_NAME, f"request_id={request_id} | error={error_msg}")
            except Exception as e:
                responses[idx] = self.create_response(status="error", message=str(e))
                app_logger.error("peanuts", f"request_id={request_id} | error={str(e)}")

        if peanut_requests:
            try:
                # processing_results = self.peanuts_process_requests(peanut_requests)
                processing_results = peanuts.process_requests(peanut_requests)
                app_logger.info("peanuts", f"Processing completed | results_count={len(processing_results)}")
                
                for idx, processing_result in enumerate(processing_results):
                    original_idx = requests_mapping[idx]
                    request_id = request_ids[original_idx]
                    start_time = start_times[original_idx]
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time if start_time else 0
                    
                    # Create response data
                    response_data = {
                        "processing_time_seconds": processing_time,
                        "status": processing_result.status,
                        "message": processing_result.message,
                        "output_xlsx_path": str(processing_result.excel_filename) if hasattr(processing_result, 'excel_filename') and processing_result.excel_filename else None
                    }
                    
                    # Save result file as artifact if it exists
                    if hasattr(processing_result, 'excel_filename') and processing_result.excel_filename:
                        excel_path = Path(processing_result.excel_filename)
                        if excel_path.exists():
                            # Read the Excel file data and save as artifact
                            with open(excel_path, 'rb') as f:
                                excel_data = f.read()
                            artifact_path = artifact_service.save_artifact("peanuts", f"result_{request_id}.xlsx", excel_data)
                            # Update response data with artifact path
                            if artifact_path:
                                response_data["output_xlsx_path"] = artifact_path
                    
                    # Save response data as artifact
                    artifact_service.save_artifact("peanuts", f"response_{request_id}.json", response_data)
                    
                    # Log response to app.log
                    status = response_data.get("status", "unknown")
                    app_logger.info(self.SERVICE_NAME, f"Response sent | request_id={request_id} | status={status} | time={processing_time:.2f}s")
                    
                    responses[original_idx] = self.create_response(
                        status=processing_result.status,
                        message=processing_result.message,
                    )
                    
            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                error_trace = traceback.format_exc()
                app_logger.error(self.SERVICE_NAME, f"error={str(e)}")
                
                # Log error for all requests that failed
                for idx, request_id in enumerate(request_ids):
                    if request_id:
                        app_logger.error("peanuts", f"Request failed | request_id={request_id} | error={str(e)}")
                
                responses[0] = self.create_response(
                    status="error", message=error_msg, error_trace=error_trace
                )

        app_logger.info("general", f"Batch processing completed | batch_size={len(requests)}")
        return responses

# peanut_deployment.py
PeanutsDeployment = Peanuts.bind()



if __name__ == "__main__":
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(PeanutsDeployment, route_prefix="/peanuts/process_image")
    
