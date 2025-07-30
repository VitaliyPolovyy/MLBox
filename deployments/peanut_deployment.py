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
from mlbox.utils.logger import get_logger

# Initialize MLBox logger
mlbox_logger = get_logger(ROOT_DIR)


@serve.deployment
class Peanuts:
    def __init__(self):
        print(">>> START")
        # Temporary directory for saving files during processing
        self.tmp_dir = ROOT_DIR / "ASSETS" / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        print(">>> PATH OK:", self.tmp_dir)
        # Load the JSON schema for peanut requests
        request_json_schema_file = (
            ROOT_DIR / "ASSETS" / "json-schemas" / "peanut-input.json"
        )
        
        with open(request_json_schema_file, "r", encoding="utf-8") as schema_file:
            self.peanut_request_json_schema = json.load(schema_file)
        print(">>> SCHEMA LOADED")

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

        peanut_requests: List[peanuts.PeanutProcessingRequest] = []
        responses = [None] * len(requests)
        request_ids = [None] * len(requests)
        start_times = [None] * len(requests)

        requests_mapping = {}

        for idx, request in enumerate(requests):
            start_time = time.time()
            request_id = None
            
            try:
                print(f"--- Request {idx}: start parsing form ---")
                form = await request.form()
                print(f"--- Request {idx}: form keys = {list(form.keys())} ---")

                image_file = form["image"]
                image_data = await image_file.read()
                print(f"--- Request {idx}: image filename = {image_file.filename}, size = {len(image_data)} bytes ---")
                
                # Save input image
                image = PILImage.open(BytesIO(image_data))
                
                # Create temporary image file for logging
                temp_image_path = self.tmp_dir / f"input_{idx}_{image_file.filename}"
                image.save(temp_image_path)

                json_body = json.loads(form["json"])
                validate(instance=json_body, schema=self.peanut_request_json_schema)
                peanut_input_json = PeanutInputJson.from_json(json.dumps(json_body)) # pylint: disable=no-member

                # Log request
                request_data = {
                    "client_ip": request.client.host if request.client else "unknown",
                    "file": image_file.filename,
                    "service_code": json_body.get("service_code"),
                    "alias": peanut_input_json.alias,
                    "key": peanut_input_json.key,
                    "response_method": peanut_input_json.response_method,
                    "response_endpoint": peanut_input_json.response_endpoint
                }
                
                request_id = mlbox_logger.log_request("peanuts", request_data)
                request_ids[idx] = request_id
                start_times[idx] = start_time
                
                # Save input image as artifact
                mlbox_logger.save_artifact(
                    service="peanuts",
                    artifact_type="images",
                    file_path=temp_image_path,
                    request_id=request_id,
                    metadata={"original_filename": image_file.filename}
                )

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
                if request_id:
                    mlbox_logger.log_error("peanuts", request_id, e, {"request_idx": idx})
            except Exception as e:
                error_msg = f"Request parsing error: {str(e)}"
                responses[idx] = self.create_response(
                    status="error", message=error_msg
                )
                if request_id:
                    mlbox_logger.log_error("peanuts", request_id, e, {"request_idx": idx, "stage": "parsing"})

        if peanut_requests:
            try:
                # processing_results = self.peanuts_process_requests(peanut_requests)
                processing_results = peanuts.process_requests(peanut_requests)
                print(f"!!! 2 len(processing_results) = {len(processing_results)}")
                
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
                            artifact_path = mlbox_logger.save_artifact(
                                service="peanuts",
                                artifact_type="results",
                                file_path=excel_path,
                                request_id=request_id,
                                metadata={"result_type": "excel_report"}
                            )
                            # Update response data with artifact path
                            if artifact_path:
                                response_data["output_xlsx_path"] = artifact_path
                    
                    # Log response
                    mlbox_logger.log_response("peanuts", request_id, response_data)
                    
                    responses[original_idx] = self.create_response(
                        status=processing_result.status,
                        message=processing_result.message,
                    )
                    
            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                error_trace = traceback.format_exc()
                
                # Log error for all requests that failed
                for idx, request_id in enumerate(request_ids):
                    if request_id:
                        mlbox_logger.log_error("peanuts", request_id, e, {
                            "request_idx": idx,
                            "stage": "processing",
                            "error_trace": error_trace
                        })
                
                responses[0] = self.create_response(
                    status="error", message=error_msg, error_trace=error_trace
                )

        return responses

# peanut_deployment.py
PeanutsDeployment = Peanuts.bind()



if __name__ == "__main__":
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(PeanutsDeployment)
    
