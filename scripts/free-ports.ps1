# Free backend (8000) and frontend (3000) ports.
#
# Why this is not trivial:
#   `uvicorn --reload` starts a reloader parent and a server child. The child
#   inherits the listening socket and is launched through multiprocessing.spawn,
#   so its command line does NOT contain "uvicorn". If the parent dies first,
#   the socket stays LISTENING but is still attributed to the dead parent PID,
#   which makes "kill the port owner" fail.
#
# Strategy:
#   1. Kill whatever currently owns the port (works in the normal case).
#   2. If a target port is still busy, kill our own project processes by
#      command line: anything running the backend venv python or the frontend vite.
#   3. Re-check and report.
#
# ASCII-only on purpose: Windows PowerShell 5.1 reads .ps1 as ANSI, so non-ASCII
# text in a UTF-8 file without BOM gets mangled and breaks parsing.

$ports = @(8000, 3000)

function Get-PortOwner {
    param([int]$Port)
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return @($conns)
}

function Test-PortsBusy {
    foreach ($p in $ports) {
        if (@(Get-PortOwner -Port $p).Count -gt 0) { return $true }
    }
    return $false
}

Write-Host "Releasing ports 8000 / 3000 ..."

# --- Step 1: kill port owners ---
for ($round = 1; $round -le 3; $round++) {
    $killed = $false
    foreach ($port in $ports) {
        foreach ($conn in @(Get-PortOwner -Port $port)) {
            $ownerId = $conn.OwningProcess
            try {
                $proc = Get-Process -Id $ownerId -ErrorAction Stop
                Write-Host ("  killing {0} (PID {1}) on port {2}" -f $proc.ProcessName, $proc.Id, $port)
                Stop-Process -Id $proc.Id -Force -ErrorAction Stop
                $killed = $true
            }
            catch {
                Write-Host ("  PID {0} already gone; socket probably inherited by a child" -f $ownerId)
            }
        }
    }
    if (-not $killed) { break }
    Start-Sleep -Milliseconds 900
}

# --- Step 2: if still busy, kill our project's python / node processes ---
if (Test-PortsBusy) {
    Write-Host "  port still busy -> matching project processes by command line"

    $procs = @()
    try {
        $procs = Get-CimInstance Win32_Process -ErrorAction Stop
    }
    catch {
        Write-Host "  WMI unavailable, falling back to Get-Process"
    }

    $victims = @()

    if ($procs.Count -gt 0) {
        foreach ($proc in $procs) {
            $cmd = $proc.CommandLine
            if (-not $cmd) { continue }

            $isOurs = $false
            # backend venv python (covers uvicorn worker / multiprocessing spawn)
            if ($cmd -like '*agentuse\backend\venv\Scripts\python.exe*') { $isOurs = $true }
            # frontend vite dev server
            if ($cmd -like '*agentuse\frontend*' -and $cmd -like '*vite*') { $isOurs = $true }

            if ($isOurs) { $victims += $proc.ProcessId }
        }
    }

    # Fallback: any python.exe alive while a backend port is busy is suspicious,
    # but we only take it when the venv python pattern matched nothing.
    if ($victims.Count -eq 0) {
        foreach ($proc in Get-Process python, node -ErrorAction SilentlyContinue) {
            if ($proc.Path -and $proc.Path -like '*agentuse*') { $victims += $proc.Id }
        }
    }

    foreach ($vid in ($victims | Select-Object -Unique)) {
        try {
            $p = Get-Process -Id $vid -ErrorAction Stop
            Write-Host ("  killing project process {0} (PID {1})" -f $p.ProcessName, $vid)
            Stop-Process -Id $vid -Force -ErrorAction Stop
        }
        catch {
            Write-Host ("  cannot kill PID {0}" -f $vid)
        }
    }

    Start-Sleep -Milliseconds 1200
}

# --- Report ---
Write-Host ""
Write-Host "Status:"
foreach ($port in $ports) {
    $listen = @(Get-PortOwner -Port $port)
    if ($listen.Count -gt 0) {
        Write-Host ("  port {0}: STILL IN USE (PID {1})" -f $port, $listen[0].OwningProcess) -ForegroundColor Yellow
    }
    else {
        Write-Host ("  port {0}: free" -f $port) -ForegroundColor Green
    }
}

if (Test-PortsBusy) {
    Write-Host ""
    Write-Host "If a port is still held, the process may need admin rights." -ForegroundColor Yellow
    Write-Host "Run this script from an elevated PowerShell, or reboot to clear the socket."
}
