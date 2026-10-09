from pathlib import Path
p = Path("ml_cv_dino_probe.py")
s = p.read_text(encoding="utf-8")
bad = (
    "            (df.get(\"source\") if \"source\" in df.columns else None)
"
    '            and df[df.split == "test"][\"source\"].unique().tolist() or []))"'
)
good = '            if "source" in df.columns else []))'
assert bad in s, "bad fragment not found"
s = s.replace(bad, good)
p.write_text(s, encoding="utf-8")
print("patched OK")
