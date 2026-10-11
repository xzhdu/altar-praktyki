import json
from pathlib import Path

# Config
SIGLIP_PATH = Path(r"C:\Users\mateu\Desktop\Embedding_SigLip2\SigLip2_Base_FixRes.json")
OPENCLIP_PATH = Path(r"C:\Users\mateu\Desktop\Embedding_OpenClip\Wyniki_ewaluacji_OpenCLIP32_multilingual.json")


def load_json(path: Path) -> list | None:
    if not path.exists():
        print(f"Blad: Nie znaleziono pliku {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Blad odczytu {path.name}: {e}")
        return None


def calculate_metrics(results_data: list) -> dict:
    total = len(results_data)
    if total == 0:
        return {}

    top1 = sum(1 for r in results_data if r.get("correct_result_position") == 1)
    top3 = sum(1 for r in results_data if r.get("correct_result_position") and r["correct_result_position"] <= 3)
    top5 = sum(1 for r in results_data if r.get("correct_result_position") and r["correct_result_position"] <= 5)

    positions = [r["correct_result_position"] for r in results_data if r.get("correct_result_position") is not None]
    avg_rank = sum(positions) / len(positions) if positions else 0.0

    avg_embed_time = sum(r.get("embedding_generation_time_ms", 0) for r in results_data) / total
    avg_search_time = sum(r.get("search_time_ms", 0) for r in results_data) / total

    return {
        "total_queries": total,
        "top1_acc": (top1 / total) * 100,
        "top3_acc": (top3 / total) * 100,
        "top5_acc": (top5 / total) * 100,
        "avg_rank": avg_rank,
        "avg_embed_time_ms": avg_embed_time,
        "avg_search_time_ms": avg_search_time
    }


def main():
    siglip_data = load_json(SIGLIP_PATH)
    openclip_data = load_json(OPENCLIP_PATH)

    if not siglip_data or not openclip_data:
        print("Nie udalo sie wczytac obu plikow JSON. Analiza przerwana.")
        return

    m_sig = calculate_metrics(siglip_data)
    m_clip = calculate_metrics(openclip_data)

    print("\n" + "=" * 70)
    print("POROWNANIE MODELI: SigLIP 2 FixRes vs Multilingual OpenCLIP")
    print("=" * 70)
    print(f"{'Metryka':<35} | {'SigLIP 2 FixRes':<15} | {'Multilingual OpenCLIP':<15}")
    print("-" * 70)
    print(f"{'Liczba zapytan':<35} | {m_sig['total_queries']:<15} | {m_clip['total_queries']:<15}")
    print(f"{'Top-1 Accuracy (%)':<35} | {m_sig['top1_acc']:.1f}%{'':<10} | {m_clip['top1_acc']:.1f}%")
    print(f"{'Top-3 Accuracy (%)':<35} | {m_sig['top3_acc']:.1f}%{'':<10} | {m_clip['top3_acc']:.1f}%")
    print(f"{'Top-5 Accuracy (%)':<35} | {m_sig['top5_acc']:.1f}%{'':<10} | {m_clip['top5_acc']:.1f}%")
    print(f"{'Srednia pozycja (im nizsza tym lepiej)':<35} | {m_sig['avg_rank']:.2f}{'':<11} | {m_clip['avg_rank']:.2f}")
    print(f"{'Sr. czas generowania tekstu (ms)':<35} | {m_sig['avg_embed_time_ms']:.2f} ms{'':<7} | {m_clip['avg_embed_time_ms']:.2f} ms")
    print(f"{'Sr. czas wyszukiwania w bazie (ms)':<35} | {m_sig['avg_search_time_ms']:.2f} ms{'':<7} | {m_clip['avg_search_time_ms']:.2f} ms")
    print("=" * 70)

    # Wskazanie wyższego wyniku Top-1
    if m_clip['top1_acc'] > m_sig['top1_acc']:
        diff = m_clip['top1_acc'] - m_sig['top1_acc']
        print(f"\nWyzsza dokladnosc: Multilingual OpenCLIP (przewaga +{diff:.1f}% w Top-1 Accuracy)")
    elif m_sig['top1_acc'] > m_clip['top1_acc']:
        diff = m_sig['top1_acc'] - m_clip['top1_acc']
        print(f"\nWyzsza dokladnosc: SigLIP 2 FixRes (przewaga +{diff:.1f}% w Top-1 Accuracy)")
    else:
        print("\nWyniki Top-1 Accuracy sa identyczne.")

    # Bezpieczne mapowanie danych po kluczu identyfikacyjnym
    siglip_lookup = {
        (item["image_id"], item["query"]): item.get("correct_result_position")
        for item in siglip_data
    }

    print("\nPRZYKLADY ZAPYTAŃ, GDZIE MULTILINGUAL OPENCLIP OSIAGNAL LEPSZA POZYCJE:")
    count = 0
    for clip_item in openclip_data:
        key = (clip_item["image_id"], clip_item["query"])
        pos_sig = siglip_lookup.get(key)
        pos_clip = clip_item.get("correct_result_position")

        if pos_clip == 1 and (pos_sig is None or pos_sig > 1):
            count += 1
            if count <= 5:
                sig_pos_str = str(pos_sig) if pos_sig is not None else "Brak"
                print(f"  * Obraz/PDF: {clip_item['image_id']}")
                print(f"    Zapytanie: '{clip_item['query']}'")
                print(f"    Pozycja SigLIP 2: {sig_pos_str} -> Pozycja OpenCLIP: {pos_clip}\n")


if __name__ == "__main__":
    main()