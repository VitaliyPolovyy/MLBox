# Fixing Cursor SSL Certificate Error on Windows

## Error: `net::ERR_CERT_AUTHORITY_INVALID`

This error occurs when Cursor can't verify the SSL certificate of the update server. Common causes:

### Solution 1: Check System Time
1. Right-click on the clock in Windows taskbar
2. Select "Adjust date/time"
3. Ensure "Set time automatically" is ON
4. Click "Sync now" if available
5. Restart Cursor

### Solution 2: Update Windows Certificates
1. Open Windows Update (Settings → Update & Security → Windows Update)
2. Install all pending updates
3. Restart your computer
4. Try updating Cursor again

### Solution 3: Corporate Proxy/Firewall
If you're behind a corporate firewall/proxy:

1. **Export corporate certificate:**
   - Ask your IT department for the corporate root CA certificate
   - Or export it from your browser (Chrome: Settings → Privacy → Security → Manage certificates)

2. **Install certificate in Windows:**
   - Press `Win + R`, type `certmgr.msc`, press Enter
   - Navigate to "Trusted Root Certification Authorities" → "Certificates"
   - Right-click → "All Tasks" → "Import"
   - Select your corporate certificate file
   - Follow the wizard

3. **Configure Cursor to use system certificates:**
   - Close Cursor completely
   - Set environment variable (if needed):
     - Press `Win + R`, type `sysdm.cpl`, press Enter
     - Go to "Advanced" tab → "Environment Variables"
     - Add `SSL_CERT_FILE` pointing to your certificate bundle (if applicable)

### Solution 4: Clear Cursor Cache (Windows)
1. Close Cursor completely
2. Press `Win + R`, type `%APPDATA%\Cursor`, press Enter
3. Delete or rename the `Cache` folder
4. Also check `%LOCALAPPDATA%\Cursor` for cache folders
5. Restart Cursor

### Solution 5: Manual Update
If automatic update fails:

1. Visit https://cursor.com/en-US/downloads
2. Download the latest Windows installer
3. Run the installer (it will update your existing installation)
4. Restart Cursor

### Solution 6: Disable SSL Verification (NOT RECOMMENDED - Last Resort)
Only use this if you understand the security implications:

1. Close Cursor
2. Create a shortcut to Cursor
3. Right-click shortcut → Properties
4. In "Target" field, add: `--ignore-certificate-errors` at the end
5. Launch Cursor from this shortcut

**Warning:** This disables SSL verification and makes you vulnerable to man-in-the-middle attacks.

### Solution 7: Check Antivirus/Firewall
1. Temporarily disable antivirus/firewall
2. Try updating Cursor
3. If it works, add Cursor to your antivirus/firewall exceptions
4. Re-enable your security software

### Solution 8: Network Configuration
1. Check if you're using a VPN - try disconnecting
2. Check proxy settings:
   - Settings → Network & Internet → Proxy
   - Ensure "Use a proxy server" matches your network requirements
3. Try a different network (mobile hotspot) to test if it's network-specific

## Most Common Fix
For most users, **Solution 1 (System Time)** or **Solution 5 (Manual Update)** resolves the issue.


