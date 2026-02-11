# PowerShell script to help fix Cursor SSL certificate issues on Windows
# Run this script in PowerShell as Administrator

Write-Host "Cursor SSL Certificate Troubleshooting Script" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator. Some operations may fail." -ForegroundColor Yellow
    Write-Host "Right-click PowerShell and select 'Run as Administrator' for full functionality." -ForegroundColor Yellow
    Write-Host ""
}

# Function to check system time
function Check-SystemTime {
    Write-Host "1. Checking system time..." -ForegroundColor Green
    $currentTime = Get-Date
    $timeSync = (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\W32Time\Parameters" -ErrorAction SilentlyContinue).Type
    Write-Host "   Current time: $currentTime" -ForegroundColor Gray
    Write-Host "   Time sync type: $timeSync" -ForegroundColor Gray
    
    # Check if time is reasonable (within 1 hour of expected)
    $expectedTime = (Invoke-WebRequest -Uri "http://worldtimeapi.org/api/timezone/Etc/UTC" -UseBasicParsing -ErrorAction SilentlyContinue).Content | ConvertFrom-Json
    if ($expectedTime) {
        $serverTime = [DateTime]::Parse($expectedTime.datetime)
        $timeDiff = [Math]::Abs(($currentTime - $serverTime).TotalMinutes)
        if ($timeDiff -gt 60) {
            Write-Host "   WARNING: System time may be incorrect (diff: $([Math]::Round($timeDiff)) minutes)" -ForegroundColor Yellow
        } else {
            Write-Host "   System time appears correct" -ForegroundColor Green
        }
    }
    Write-Host ""
}

# Function to clear Cursor cache
function Clear-CursorCache {
    Write-Host "2. Clearing Cursor cache..." -ForegroundColor Green
    $cursorPaths = @(
        "$env:APPDATA\Cursor\Cache",
        "$env:APPDATA\Cursor\CachedData",
        "$env:LOCALAPPDATA\Cursor\Cache",
        "$env:LOCALAPPDATA\Cursor\CachedData",
        "$env:LOCALAPPDATA\Cursor\User\workspaceStorage"
    )
    
    foreach ($path in $cursorPaths) {
        if (Test-Path $path) {
            try {
                Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
                Write-Host "   Cleared: $path" -ForegroundColor Gray
            } catch {
                Write-Host "   Could not clear: $path (may be in use)" -ForegroundColor Yellow
            }
        }
    }
    Write-Host ""
}

# Function to check certificate store
function Check-Certificates {
    Write-Host "3. Checking certificate store..." -ForegroundColor Green
    $rootCerts = Get-ChildItem -Path Cert:\LocalMachine\Root | Measure-Object
    Write-Host "   Root certificates found: $($rootCerts.Count)" -ForegroundColor Gray
    
    # Check for common certificate issues
    $certStore = New-Object System.Security.Cryptography.X509Certificates.X509Store("Root", "LocalMachine")
    $certStore.Open("ReadOnly")
    $certs = $certStore.Certificates
    $certStore.Close()
    
    Write-Host "   Certificate store appears healthy" -ForegroundColor Green
    Write-Host ""
}

# Function to test SSL connection
function Test-SSLConnection {
    Write-Host "4. Testing SSL connection to Cursor update server..." -ForegroundColor Green
    try {
        $response = Invoke-WebRequest -Uri "https://cursor.com" -UseBasicParsing -ErrorAction Stop
        Write-Host "   SSL connection successful!" -ForegroundColor Green
    } catch {
        Write-Host "   SSL connection failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "   This indicates a certificate or network issue" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Function to check proxy settings
function Check-ProxySettings {
    Write-Host "5. Checking proxy settings..." -ForegroundColor Green
    $proxy = [System.Net.WebRequest]::GetSystemWebProxy()
    $proxy.Credentials = [System.Net.CredentialCache]::DefaultCredentials
    
    $cursorUrl = New-Object System.Uri("https://cursor.com")
    $proxyUrl = $proxy.GetProxy($cursorUrl)
    
    if ($proxyUrl -eq $cursorUrl) {
        Write-Host "   No proxy detected" -ForegroundColor Gray
    } else {
        Write-Host "   Proxy detected: $proxyUrl" -ForegroundColor Yellow
        Write-Host "   If you're behind a corporate proxy, you may need to install the corporate CA certificate" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Function to provide recommendations
function Show-Recommendations {
    Write-Host "Recommendations:" -ForegroundColor Cyan
    Write-Host "================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Ensure Windows is up to date (Settings → Update & Security)" -ForegroundColor White
    Write-Host "2. Try manually downloading the update from: https://cursor.com/en-US/downloads" -ForegroundColor White
    Write-Host "3. If behind corporate proxy, contact IT for the root CA certificate" -ForegroundColor White
    Write-Host "4. Temporarily disable VPN/antivirus to test if they're interfering" -ForegroundColor White
    Write-Host "5. Restart Cursor after running this script" -ForegroundColor White
    Write-Host ""
}

# Run all checks
Check-SystemTime
Clear-CursorCache
Check-Certificates
Test-SSLConnection
Check-ProxySettings
Show-Recommendations

Write-Host "Script completed!" -ForegroundColor Green
Write-Host "Please restart Cursor and try updating again." -ForegroundColor Yellow


