@echo off
echo Checking what happened during Rust builds...
echo.

echo ========================================
echo Check if build log files exist
echo ========================================
docker run --rm chef-genius-ai-chef-genius-training:latest ls -la /tmp/

echo.
echo ========================================
echo Check core build log
echo ========================================
docker run --rm chef-genius-ai-chef-genius-training:latest cat /tmp/core_build.log

echo.
echo ========================================
echo Check dataloader build log  
echo ========================================
docker run --rm chef-genius-ai-chef-genius-training:latest cat /tmp/dataloader_build.log

echo.
echo ========================================
echo Check if Rust files exist
echo ========================================
docker run --rm chef-genius-ai-chef-genius-training:latest ls -la chef_genius_core/
docker run --rm chef-genius-ai-chef-genius-training:latest ls -la cli/rust_dataloader/

echo.
echo ========================================
echo Check site-packages for any chef modules
echo ========================================
docker run --rm chef-genius-ai-chef-genius-training:latest find /opt/conda/lib/python3.11/site-packages -name "*chef*" -o -name "*genius*"

pause