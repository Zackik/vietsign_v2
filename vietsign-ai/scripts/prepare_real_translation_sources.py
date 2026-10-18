from __future__ import annotations

import json

from vietsign.translation.real_sources import prepare_real_translation_sources

if __name__ == "__main__":
    print(json.dumps(prepare_real_translation_sources(), indent=2, ensure_ascii=False))
