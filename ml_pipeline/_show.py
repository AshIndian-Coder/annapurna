from pathlib import Path
p = Path("ml_cv_dino_probe.py")
s = p.read_text(encoding="utf-8")
i = s.find("test_sources")
print("=== context around test_sources ===")
print(repr(s[i-90:i+300]))
print()
print("bad fragment present:", "or []" in s and "test_sources" in s)
