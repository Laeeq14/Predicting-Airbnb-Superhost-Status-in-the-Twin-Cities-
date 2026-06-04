# Superhost Predictor & Performance Simulator
# One-command launcher: trains model (if needed) + starts FastAPI
import subprocess, sys, os
from pathlib import Path

ROOT         = Path(__file__).parent
BEST_MODEL   = ROOT / 'ml_pipeline' / 'best_model.joblib'
XGB_MODEL    = ROOT / 'ml_pipeline' / 'xgb_model.joblib'   # legacy fallback
PYTHON       = sys.executable   # always uses the Python that ran this script

def main():
    print("=" * 55)
    print("  Superhost Predictor & Performance Simulator")
    print("=" * 55)
    print(f"  Python: {PYTHON}")

    # Check for any trained model
    model_exists = BEST_MODEL.exists() or XGB_MODEL.exists()

    if not model_exists:
        print("\n[*] No trained model found. Starting training...")
        print("    This may take 15-30 minutes with extended model set.")
        print("    Models: RF, XGBoost, LightGBM, CatBoost, Ensemble\n")
        result = subprocess.run(
            [PYTHON, str(ROOT / 'ml_pipeline' / 'train_model.py')],
            cwd=str(ROOT)
        )
        if result.returncode != 0:
            print("\n[ERROR] Training failed. Check output above.")
            sys.exit(1)
        print("\n[*] Training complete!")
    else:
        if BEST_MODEL.exists():
            print(f"\n[*] Using best model: {BEST_MODEL.name}")
        else:
            print(f"\n[*] Using legacy model: {XGB_MODEL.name}")
            print("    Tip: re-run train_model.py to get the extended model set.")

    print("\n[*] Starting web server...")
    print("    URL: http://localhost:8000")
    print("    Press Ctrl+C to stop\n")

    os.chdir(ROOT)
    subprocess.run([
        PYTHON, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ], cwd=str(ROOT))

if __name__ == '__main__':
    main()
