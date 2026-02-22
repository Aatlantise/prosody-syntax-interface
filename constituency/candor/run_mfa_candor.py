from pathlib import Path
import subprocess
import os


def main():
    corpus_base = Path("/home/jm3743/data/candor/mfa/pre_alignment")
    outputs_base = Path("/home/jm3743/data/candor/mfa/post_alignment")

    os.makedirs(outputs_base, exist_ok=True)

    ## Perform MFA
    command = [
        "mfa",
        "align",
        os.path.abspath(corpus_base),
        "english_us_arpa",
        "english_us_arpa",
        os.path.abspath(outputs_base),
        "--clean",
        "--use_mp",
        "--overwrite",
        "--textgrid_cleanup",
        "--output_format",
        "csv",
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"Successfully aligned")
    except subprocess.CalledProcessError as e:
        print(f"Error aligning: {e}")


if __name__ == "__main__":
    main()
