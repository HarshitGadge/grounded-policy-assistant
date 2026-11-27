"""Print what eCFR's XML for a part looks like: element counts and one section's raw XML.

Used by the live CI job so a change in eCFR's format shows up in the log.

    python scripts/inspect_ecfr.py data/corpus/raw/title-12-part-1005-2026-11-20.xml 1005.6
"""

import sys
import xml.etree.ElementTree as ET
from collections import Counter


def main(path: str, section: str | None = None, limit: int = 4000) -> None:
    root = ET.parse(path).getroot()
    print(f"root <{root.tag} {dict(root.attrib)}>")
    print("elements:", dict(Counter(el.tag for el in root.iter()).most_common()))
    print("div types:", dict(Counter(el.get("TYPE") for el in root.iter() if el.get("TYPE"))))
    if section:
        for el in root.iter():
            if el.get("N") == section or (el.get("N") or "").startswith(section + " "):
                print(f"\n--- {section} (first {limit} chars) ---")
                print(ET.tostring(el, encoding="unicode")[:limit])
                break
        else:
            print(f"\nno element with N={section!r}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
