"""One-command reproduction after environment setup; completed runs are resumed."""
import os
import subprocess
import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MPLCONFIGDIR=str(root / "tmp/matplotlib"))
    for module in ["src.validate", "src.run_experiments", "src.run_controls", "src.diagnose_oll",
                   "src.analyze", "src.write_log", "src.verify_results"]:
        subprocess.run([sys.executable, "-m", module], env=env, check=True)
    compiler = root / "tools/tectonic/tectonic"
    if not compiler.exists():
        raise RuntimeError("Install Tectonic or restore tools/tectonic/tectonic; see README.md")
    env["XDG_CACHE_HOME"] = str(root / "tmp/tex-cache")
    subprocess.run([str(compiler), "--keep-logs", "--keep-intermediates", "--outdir", "paper", "paper/paper.tex"],
                   env=env, check=True)
    print("Reproduction complete: paper/paper.pdf")

if __name__ == "__main__":
    main()
