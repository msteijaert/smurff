# Sample script to install anaconda under windows
# Authors: Stuart Mumford
# Borrwed from: Olivier Grisel and Kyle Kastner
# License: BSD 3 clause

Param (
    [string]$python_version = "3", 
    [string]$conda_version = "latest", 
    [string]$platform = "x86_64",    
    [string]$destination = $ENV:CONDA
)

$MINICONDA_URL = "http://repo.continuum.io/miniconda/"

function DownloadMiniconda ($python_version, $conda_version, $pdirlatform_suffix) {
    $webclient = New-Object System.Net.WebClient
    $filename = "Miniconda" + $python_version + "-" + $conda_version + "-Windows-" + $platform_suffix + ".exe"

    $url = $MINICONDA_URL + $filename

    $basedir = $pwd.Path + "\"
    $filepath = $basedir + $filename
    if (Test-Path $filename) {
        Write-Host "Reusing" $filepath
        return $filepath
    }

    # Download and retry up to 3 times in case of network transient errors.
    Write-Host "Downloading" $filename "from" $url
    $retry_attempts = 2
    for($i=0; $i -lt $retry_attempts; $i++){
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
   }
   if (Test-Path $filepath) {
       Write-Host "File saved at" $filepath
   } else {
       # Retry once to get the error message if any at the last try
       $webclient.DownloadFile($url, $filepath)
   }
   return $filepath
}

function InstallMiniconda ($python_version, $conda_version, $architecture, $destination) {
    Write-Host "Installing miniconda" $python_version "for" $architecture "bit architecture to" $destination
    if (Test-Path $destination) {
        Write-Host $destination "already exists, skipping."
        return $false
    }
    if ($architecture -eq "x86") {
        $platform_suffix = "x86"
    } else {
        $platform_suffix = "x86_64"
    }
    $filepath = DownloadMiniconda $python_version $conda_version $platform_suffix
    Write-Host "Installing" $filepath "to" $destination
    $args = "/NoRegistry=1 /S /D=" + $destination
    Write-Host $filepath $args
    Start-Process -FilePath $filepath -ArgumentList $args -Wait -Passthru
    #Start-Sleep -s 15
    if (Test-Path $destination) {
        Write-Host "Miniconda $python_version ($architecture) installation complete"
    } else {
        Write-Host "Failed to install MiniConda in $destination"
        Exit 1
    }
}


function main () {
    InstallMiniconda $python_version $conda_version $platform $destination
}

main