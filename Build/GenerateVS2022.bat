@echo off
setlocal
cmake -S "%~dp0." -B "%~dp0VS2022" -G "Visual Studio 17 2022" -A x64
exit /b %errorlevel%
