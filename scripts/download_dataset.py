#!/usr/bin/env python3
"""
Script generico per il download di dataset.
Utilizzo:
    python scripts/download_dataset.py https://physionet.org/files/ptb-xl/1.0.3/
"""

import argparse
import subprocess
import sys
from pathlib import Path

def check_wget():
    """Verifica se wget è installato nel sistema."""
    try:
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_dataset(url, output_path):
    """
    Scarica il dataset utilizzando wget (raccomandato per PhysioNet).
    PhysioNet richiede spesso il mirroring (-m) e il no-parent (-np).
    """
    print(f"🚀 Inizio download da: {url}")
    print(f"📂 Destinazione: {output_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)

    # Comando wget ottimizzato per PhysioNet
    # -m: mirror, -np: no-parent (non risale alle cartelle superiori)
    # -nH: no-host-directories, --cut-dirs: evita di creare catene di sottocartelle inutili
    # -P: specifica la cartella di output
    cmd = [
        "wget", "-m", "-np", "-nH",
        "--cut-dirs=3", # Regola questo numero per tagliare i segmenti dell'URL nel path locale
        "-P", str(output_path),
        url
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Download completato con successo in {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Errore durante il download: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download generico per PhysioNet")
    # Rendiamo l'URL un argomento obbligatorio
    parser.add_argument("url", help="URL del dataset su PhysioNet (es. https://physionet.org/files/ptb-xl/1.0.3/)")
    # Default su data/raw come concordato
    parser.add_argument("--output", default="data/raw", help="Cartella di destinazione (default: data/raw)")

    args = parser.parse_args()

    if not check_wget():
        print("❌ Errore: 'wget' non è installato. Per favore installalo (es. 'brew install wget' o 'sudo apt install wget').")
        sys.exit(1)

    # Esegui il download
    success = download_dataset(args.url, Path(args.output))
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()