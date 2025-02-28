import json
import sys
import traceback
from io import BytesIO
from typing import List

import ray
from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate

# from datetime import datetime
from loguru import logger
from PIL import Image as PILImage
from ray import serve
from starlette.requests import Request
from mlbox.services.peanuts.datatype import PeanutInputJson
from mlbox.services.peanuts import peanuts
from mlbox.settings import ROOT_DIR

logger.remove()  # Remove default logger
# logger.add(sys.stdout, level="INFO", enqueue=True)
# logger.add(f'{__name__}__run.log', filter=__name__ , rotation='1 week')
logger.add(sys.stdout, level="INFO", enqueue=False)  # No multiprocessing queue


@serve.deployment
class Peanuts:
    def __init__(self):
        # Temporary directory for saving files during processing
        self.tmp_dir = ROOT_DIR / "ASSETS" / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        # Load the JSON schema for peanut requests
        request_json_schema_file = (
            ROOT_DIR / "ASSETS" / "json-schemas" / "peanut-input.json"
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

        peanut_requests: List[peanuts.PeanutProcessingRequest] = []
        responses = [None] * len(requests)

        requests_mapping = {}

        for idx, request in enumerate(requests):
            try:
                form = await request.form()

                image_file = form["image"]
                image_data = await image_file.read()
                image = PILImage.open(BytesIO(image_data))

                json_body = json.loads(form["json"])
                validate(instance=json_body, schema=self.peanut_request_json_schema)
                peanut_input_json = PeanutInputJson.from_json( json.dumps(json_body)) # pylint: disable=no-member

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
                responses[idx] = self.create_response(
                    status="error", message=f"JSON schema validation error: {str(e)}"
                )
            # except Exception as e:
            #   responses[idx] = self.create_response(status = "error1", message = str(e))

        if peanut_requests:
            try:
                # processing_results = self.peanuts_process_requests(peanut_requests)
                processing_results = peanuts.process_requests(peanut_requests)
                print(f"!!! 2 len(processing_results) = {len(processing_results)}")
                for idx, processing_result in enumerate(processing_results):
                    responses[requests_mapping[idx]] = self.create_response(
                        status=processing_result.status,
                        message=processing_result.message,
                    )
            except Exception as e:
                responses[0] = self.create_response(
                    status="error", message=str(e), error_trace=traceback.format_exc()
                )

        return responses


# Expose the deployment instance (bind it)
PeanutsDeployment = Peanuts.bind()

if __name__ == "__main__":
    ray.init()
    serve.start()
    serve.run(PeanutsDeployment)
