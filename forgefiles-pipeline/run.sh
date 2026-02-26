#!/bin/bash
# ForgeFiles Pipeline — Quick Commands
# ======================================
# Usage: ./run.sh <command> [args]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BLENDER="${BLENDER_PATH:-blender}"
RENDER_SCRIPT="$SCRIPT_DIR/scripts/render_engine.py"
ORCHESTRATOR="$SCRIPT_DIR/scripts/orchestrator.py"

case "$1" in
    setup)
        python3 "$SCRIPT_DIR/scripts/setup.py" --generate-assets
        ;;

    check)
        python3 "$SCRIPT_DIR/scripts/setup.py" --check-only
        ;;

    render)
        # Quick turntable: ./run.sh render model.stl [material] [preset]
        STL="$2"
        MAT="${3:-gray_pla}"
        PRESET="${4:-portfolio}"
        $BLENDER -b --python "$RENDER_SCRIPT" -- \
            --input "$STL" --mode turntable --material "$MAT" --preset "$PRESET" \
            --platform wide vertical square --output "$SCRIPT_DIR/output/renders"
        ;;

    render-fast)
        # Fast EEVEE preview: ./run.sh render-fast model.stl
        $BLENDER -b --python "$RENDER_SCRIPT" -- \
            --input "$2" --mode turntable --fast \
            --platform wide --output "$SCRIPT_DIR/output/renders"
        ;;

    render-all)
        # Full render suite: ./run.sh render-all model.stl [preset]
        PRESET="${3:-portfolio}"
        $BLENDER -b --python "$RENDER_SCRIPT" -- \
            --input "$2" --mode all --preset "$PRESET" \
            --platform wide vertical square --output "$SCRIPT_DIR/output/renders"
        ;;

    render-ultra)
        # Maximum quality: ./run.sh render-ultra model.stl
        $BLENDER -b --python "$RENDER_SCRIPT" -- \
            --input "$2" --mode all --preset ultra \
            --platform wide vertical square --output "$SCRIPT_DIR/output/renders"
        ;;

    pipeline)
        # Full pipeline: ./run.sh pipeline model.stl
        python3 "$ORCHESTRATOR" --stl "$2" --all-platforms --output "$SCRIPT_DIR/output"
        ;;

    pipeline-fast)
        # Fast pipeline: ./run.sh pipeline-fast model.stl
        python3 "$ORCHESTRATOR" --stl "$2" --all-platforms --fast --output "$SCRIPT_DIR/output"
        ;;

    pipeline-batch)
        # Batch all STLs: ./run.sh pipeline-batch ./models/
        python3 "$ORCHESTRATOR" --stl "$2" --batch --all-platforms --output "$SCRIPT_DIR/output"
        ;;

    pipeline-ultra)
        # Ultra quality pipeline: ./run.sh pipeline-ultra model.stl
        python3 "$ORCHESTRATOR" --stl "$2" --all-platforms --preset ultra --output "$SCRIPT_DIR/output"
        ;;

    analyze)
        # Analyze STL: ./run.sh analyze model.stl
        python3 "$SCRIPT_DIR/scripts/stl_analyzer.py" "$2"
        ;;

    captions)
        # Generate captions only: ./run.sh captions "Model Name"
        python3 "$SCRIPT_DIR/scripts/caption_engine.py" "$2"
        ;;

    brand)
        # Generate brand assets: ./run.sh brand
        python3 "$SCRIPT_DIR/scripts/brand_generator.py" --all
        ;;

    *)
        echo "ForgeFiles Pipeline"
        echo "==================="
        echo ""
        echo "Usage: ./run.sh <command> [args]"
        echo ""
        echo "Setup:"
        echo "  setup                  Validate env + generate brand assets"
        echo "  check                  Check tools only"
        echo "  brand                  Generate fallback brand assets"
        echo ""
        echo "Render:"
        echo "  render <stl> [mat]     Quick turntable render"
        echo "  render-fast <stl>      Fast EEVEE preview"
        echo "  render-all <stl>       Full render suite"
        echo "  render-ultra <stl>     Maximum quality render"
        echo ""
        echo "Pipeline:"
        echo "  pipeline <stl>         Full pipeline (render+compose+captions+thumbnails)"
        echo "  pipeline-fast <stl>    Fast pipeline for testing"
        echo "  pipeline-batch <dir>   Batch all STLs in directory"
        echo "  pipeline-ultra <stl>   Ultra quality pipeline"
        echo ""
        echo "Tools:"
        echo "  analyze <stl>          Analyze STL geometry + print settings"
        echo "  captions <name>        Generate caption variants"
        echo ""
        echo "Environment:"
        echo "  BLENDER_PATH           Path to Blender (default: 'blender')"
        ;;
esac
