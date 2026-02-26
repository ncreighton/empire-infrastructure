@echo off
REM ForgeFiles Pipeline - Windows Quick Commands
REM ==============================================
REM Usage: run.bat <command> [args]

SET SCRIPT_DIR=%~dp0
SET BLENDER=blender
IF DEFINED BLENDER_PATH SET BLENDER=%BLENDER_PATH%
SET RENDER_SCRIPT=%SCRIPT_DIR%scripts\render_engine.py
SET ORCHESTRATOR=%SCRIPT_DIR%scripts\orchestrator.py

IF "%1"=="setup" (
    python "%SCRIPT_DIR%scripts\setup.py" --generate-assets
    GOTO :EOF
)

IF "%1"=="check" (
    python "%SCRIPT_DIR%scripts\setup.py" --check-only
    GOTO :EOF
)

IF "%1"=="render" (
    SET MAT=%3
    IF "%3"=="" SET MAT=gray_pla
    %BLENDER% -b --python "%RENDER_SCRIPT%" -- --input "%2" --mode turntable --material %MAT% --platform wide vertical square --output "%SCRIPT_DIR%output\renders"
    GOTO :EOF
)

IF "%1"=="render-fast" (
    %BLENDER% -b --python "%RENDER_SCRIPT%" -- --input "%2" --mode turntable --fast --platform wide --output "%SCRIPT_DIR%output\renders"
    GOTO :EOF
)

IF "%1"=="render-all" (
    %BLENDER% -b --python "%RENDER_SCRIPT%" -- --input "%2" --mode all --platform wide vertical square --output "%SCRIPT_DIR%output\renders"
    GOTO :EOF
)

IF "%1"=="render-ultra" (
    %BLENDER% -b --python "%RENDER_SCRIPT%" -- --input "%2" --mode all --preset ultra --platform wide vertical square --output "%SCRIPT_DIR%output\renders"
    GOTO :EOF
)

IF "%1"=="pipeline" (
    python "%ORCHESTRATOR%" --stl "%2" --all-platforms --output "%SCRIPT_DIR%output"
    GOTO :EOF
)

IF "%1"=="pipeline-fast" (
    python "%ORCHESTRATOR%" --stl "%2" --all-platforms --fast --output "%SCRIPT_DIR%output"
    GOTO :EOF
)

IF "%1"=="pipeline-batch" (
    python "%ORCHESTRATOR%" --stl "%2" --batch --all-platforms --output "%SCRIPT_DIR%output"
    GOTO :EOF
)

IF "%1"=="pipeline-ultra" (
    python "%ORCHESTRATOR%" --stl "%2" --all-platforms --preset ultra --output "%SCRIPT_DIR%output"
    GOTO :EOF
)

IF "%1"=="analyze" (
    python "%SCRIPT_DIR%scripts\stl_analyzer.py" "%2"
    GOTO :EOF
)

IF "%1"=="captions" (
    python "%SCRIPT_DIR%scripts\caption_engine.py" "%2"
    GOTO :EOF
)

IF "%1"=="brand" (
    python "%SCRIPT_DIR%scripts\brand_generator.py" --all
    GOTO :EOF
)

echo ForgeFiles Pipeline
echo ====================
echo.
echo Usage: run.bat ^<command^> [args]
echo.
echo Setup:
echo   setup                  Validate env + generate brand assets
echo   check                  Check tools only
echo   brand                  Generate fallback brand assets
echo.
echo Render:
echo   render ^<stl^> [mat]     Quick turntable render
echo   render-fast ^<stl^>      Fast EEVEE preview
echo   render-all ^<stl^>       Full render suite
echo   render-ultra ^<stl^>     Maximum quality render
echo.
echo Pipeline:
echo   pipeline ^<stl^>         Full pipeline
echo   pipeline-fast ^<stl^>    Fast pipeline for testing
echo   pipeline-batch ^<dir^>   Batch all STLs
echo   pipeline-ultra ^<stl^>   Ultra quality pipeline
echo.
echo Tools:
echo   analyze ^<stl^>          Analyze STL geometry
echo   captions ^<name^>        Generate caption variants
echo.
echo Set BLENDER_PATH if Blender isn't in PATH
