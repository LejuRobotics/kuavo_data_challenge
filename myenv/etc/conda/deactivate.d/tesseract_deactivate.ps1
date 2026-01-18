try { Remove-Item env:TESSDATA_PREFIX}
catch { Write-Host "Failed to execute Remove-Item env:TESSDATA_PREFIX" }
