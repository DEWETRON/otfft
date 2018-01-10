@echo off

set OTFFT_LIB_PATH=build
set OUTPUT_LIB_PATH=%2
set OUTPUT_INC_PATH=%3

if "%1"=="x64" (
  echo 64bit build selected
  set OTFFT_LIB_PATH="build64"
  set OUTPUT_LIB_PATH="%OUTPUT_LIB_PATH%64"
) else (
  echo 32bit build selected
)

set OUTPUT_LIB_PATH_DEBUG="%OUTPUT_LIB_PATH%\shared\Debug"
set OUTPUT_LIB_PATH_RELEASE="%OUTPUT_LIB_PATH%\shared\Release"

copy "%OTFFT_LIB_PATH%\Debug\otfft.lib" "%OUTPUT_LIB_PATH_DEBUG%\otfftd.lib"
copy "%OTFFT_LIB_PATH%\Debug\*.pdb" "%OUTPUT_LIB_PATH_DEBUG%"
copy "%OTFFT_LIB_PATH%\Release\otfft.lib" "%OUTPUT_LIB_PATH_RELEASE%\otfft.lib"
copy "%OTFFT_LIB_PATH%\Release\*.pdb" "%OUTPUT_LIB_PATH_RELEASE%"

copy "inc\*.h" "%OUTPUT_INC_PATH%\otfft\include"
