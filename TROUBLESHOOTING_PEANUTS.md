# Peanuts Service Connection Troubleshooting

## Issues Identified and Fixed

### ✅ Issue 1: Port Mapping Mismatch (FIXED)
**Problem**: Docker was mapping `-p 8001:8000` (host port 8001 → container port 8000), but the service inside the container was configured to listen on port 8001.

**Fix Applied**: Changed `ray_serv.yaml` to use port 8000 instead of 8001, matching the Docker port mapping.

**Action Required**: 
- Rebuild and restart your Docker container after pulling the updated configuration
- The port mapping `-p 8001:8000` is now correct

### ✅ Issue 2: Route Prefix (FIXED)
**Problem**: Route prefix in YAML was `/peanuts` but endpoint needs `/peanuts/process_image`.

**Fix Applied**: Updated `ray_serv.yaml` route_prefix to `/peanuts/process_image`.

### ⚠️ Issue 3: IP Address Mismatch (REQUIRES YOUR ACTION)
**Problem**: Your socket test uses `10.11.122.223` but your HTTP request uses `10.11.122.233`.

**Action Required**: 
1. Verify which IP address is correct for your prod server
2. Update your C# code to use the correct IP address:

```csharp
// Change this line in your C# code:
SERVER_HOST = "http://10.11.122.233:8001/peanuts/process_image";

// To match your working socket test:
SERVER_HOST = "http://10.11.122.223:8001/peanuts/process_image";
```

## Verification Steps

### 1. Check Docker Container Status
```bash
docker ps | grep mlbox
docker logs mlbox
```

### 2. Test Connection from ERP Server
From your ERP server, test the connection:

```python
import socket
import requests

# Test 1: Socket connection (this should work)
host = "10.11.122.223"  # Use the correct IP
port = 8001
socket.create_connection((host, port), timeout=5)
print("✅ Socket connection successful")

# Test 2: HTTP connection
url = f"http://{host}:8001/peanuts/process_image"
response = requests.get(url, timeout=5)  # This might return 405 Method Not Allowed, which is OK
print(f"✅ HTTP connection successful: {response.status_code}")
```

### 3. Test from Prod Server (if accessible)
```bash
# Inside the Docker container or on prod server
curl -X POST http://localhost:8000/peanuts/process_image \
  -F "image=@/path/to/test.jpg" \
  -F 'json={"alias":"TEST","key":"123","response_method":"HTTP_POST_REQUEST","response_endpoint":"http://test.com"}'
```

### 4. Check Firewall Rules
Ensure port 8001 is open on the prod server:
```bash
# On prod server
sudo ufw status
sudo netstat -tuln | grep 8001
```

## Network Architecture

```
ERP Server (Internal Network)
    ↓
    Can connect to Prod Server
    ↓
Prod Server (DMZ) - Docker Container
    ├─ Host port 8001
    └─ Container port 8000 (Ray Serve)
```

## Docker Deployment Command

Make sure you're using the correct port mapping:
```bash
docker run -d --name mlbox \
  -p 8001:8000 -p 8265:8265 \
  --env-file /path/to/.env \
  -v /path/to/artifacts:/app/artifacts \
  -v /path/to/tmp:/app/tmp \
  -v /path/to/logs:/app/logs \
  mlbox:latest
```

## Common Issues

### Connection Timeout
- **Check IP address**: Ensure you're using the correct prod server IP
- **Check port**: Verify Docker is mapping port 8001 correctly
- **Check firewall**: Ensure port 8001 is open on prod server
- **Check Docker logs**: `docker logs mlbox` to see if service started correctly

### 404 Not Found
- **Check route**: Ensure you're using `/peanuts/process_image` (not just `/peanuts`)
- **Check Ray Serve status**: The service might not have started correctly

### 500 Internal Server Error
- **Check Docker logs**: `docker logs mlbox` for detailed error messages
- **Check file permissions**: Ensure volumes are mounted correctly
- **Check environment variables**: Ensure HF_TOKEN and other required vars are set

## Next Steps

1. **Update your C# code** with the correct IP address (10.11.122.223)
2. **Rebuild Docker image** if you pulled the updated `ray_serv.yaml`:
   ```bash
   docker build -t mlbox:latest .
   ```
3. **Restart the container**:
   ```bash
   docker stop mlbox
   docker rm mlbox
   docker run -d --name mlbox -p 8001:8000 -p 8265:8265 ... mlbox:latest
   ```
4. **Test the connection** from ERP server using the updated IP address

