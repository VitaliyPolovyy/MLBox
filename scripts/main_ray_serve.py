from ray import serve
import ray
from starlette.requests import Request
from starlette.responses import JSONResponse
from mlbox.services.peanuts import peanuts
from mlbox.settings import ROOT_DIR
from pathlib import Path

@serve.deployment(route_prefix="/peanuts")
class Peanuts:
    def __init__(self):
        # Temporary directory for saving files during processing
        self.tmp_dir = ROOT_DIR / "ASSETS" / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    async def __call__(self, request: Request):
        # Route based on path suffix
        path = request.url.path

        # Check for specific endpoints
        if path.endswith("/process_image"):
            return await self.process_image(request)
        else:
            return JSONResponse({"status": "error", "message": "Invalid endpoint"}, status_code=404)

    @serve.batch(max_batch_size=10, batch_wait_timeout_s = 0)
    async def process_image(self, batched_requests):
        results = []
        for request in batched_requests:
            try:
                # Parse the form data
                form = await request.form()
                image_file = form["image"]
                image_data = await image_file.read()
                alias = form.get("alias")
                alias_key = form.get("alias_key")
                filename = image_file.filename

                # Save the file locally
                save_path = self.tmp_dir / filename
                with open(save_path, "wb") as f:
                    f.write(image_data)

                # Processing logic
                result = peanuts.process_image(save_path, alias=alias, alias_key=alias_key)

                #results.append({"filename": filename, "result": result, "alias": alias, "alias_key": alias_key})
                results.append({"filename": filename, "result": {"error": '34'}, "alias": alias, "alias_key": alias_key})
            except Exception as e:
                results.append({"filename": filename, "result": {"error": str(e)}, "alias": alias, "alias_key": alias_key})

        return results

# Expose the deployment instance (bind it)
PeanutsDeployment = Peanuts.bind()

if __name__ == "__main__":
    ray.init()
    serve.start()
    serve.run(PeanutsDeployment)
