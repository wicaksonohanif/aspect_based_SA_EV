import json

with open("notebooks/04_kaggle_training_indoroberta_vs_xlmroberta.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/nb04_content.txt", "w", encoding="utf-8") as out:
    for i, c in enumerate(nb["cells"]):
        out.write(f"=== CELL {i} ({c['cell_type']}) ===\n")
        out.write("".join(c["source"]))
        out.write("\n" + "-" * 50 + "\n")

print("Dumped nb04 to scratch/nb04_content.txt")
