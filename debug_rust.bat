@echo off
echo Testing Rust imports in Docker container...
echo.

echo ========================================
echo Test 1: Direct chef_genius_dataloader import
echo ========================================
docker run --rm chef-genius-ai-chef-genius-training:latest python -c "try: import chef_genius_dataloader; print('SUCCESS: chef_genius_dataloader imported'); print('Available functions:', [x for x in dir(chef_genius_dataloader) if not x.startswith('_')]); except Exception as e: print('FAILED:', e)"

echo.
echo ========================================
echo Test 2: Check RUST_AVAILABLE flag
echo ========================================
docker run --rm chef-genius-ai-chef-genius-training:latest bash -c "cd cli && python -c 'from fast_dataloader import RUST_AVAILABLE; print(\"RUST_AVAILABLE:\", RUST_AVAILABLE)'"

echo.
echo ========================================
echo Test 3: Check installed packages
echo ========================================
docker run --rm chef-genius-ai-chef-genius-training:latest pip list | findstr /i chef

echo.
echo ========================================
echo Test 4: Run comprehensive test
echo ========================================
docker run --rm chef-genius-ai-chef-genius-training:latest python test_rust_import.py

echo.
echo Testing complete!
pause