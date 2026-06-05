from backend.models import content, userbased, ials, bpr

_MODELS = [("content", content), ("userbased", userbased), ("ials", ials), ("bpr", bpr)]


def load_all() -> None:
    for name, mod in _MODELS:
        try:
            mod.load()
        except FileNotFoundError as e:
            print(f"[{name}] WARNING: artefact manquant — {e}")
            print(f"[{name}] Ce modèle retournera [] et le fallback popularité sera utilisé.")
        except Exception as e:
            print(f"[{name}] WARNING: échec du chargement — {e}")
