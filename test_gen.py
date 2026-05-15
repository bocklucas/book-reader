from pathlib import Path
from src.script_generate import generate_single_script_sync
import asyncio

# The chapter index is 5. We can just list the directory to confirm.
# But wait, local/output/mock_book has chapters. Let's use it.
output_dir = Path("local/output/mock_book")
# chapter 5 is index 5
script_path = generate_single_script_sync(output_dir, 5)
print(f"Generated {script_path}")
