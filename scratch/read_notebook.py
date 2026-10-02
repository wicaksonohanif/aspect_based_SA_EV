import json

with open("notebooks/03_eda_labeled_dataset.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/notebook_content.txt", "w", encoding="utf-8") as out:
    for i, c in enumerate(nb["cells"]):
        out.write(f"=== CELL {i} ({c['cell_type']}) ===\n")
        out.write("".join(c["source"]))
        out.write("\n" + "-" * 50 + "\n")

print("Saved to scratch/notebook_content.txt")
