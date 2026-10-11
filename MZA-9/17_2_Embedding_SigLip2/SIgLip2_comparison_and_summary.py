import json
from pathlib import Path

# Ścieżki do Twoich wyników
FIXRES_PATH = Path(r"C:\Users\mateu\Desktop\Embedding_SigLip2\SigLip2_Base_FixRes.json")
NAFLEX_PATH = Path(r"C:\Users\mateu\Desktop\Embedding_SigLip2\SigLip2_Base_NaFlex.json")


def load_json(path: Path):
    if not path.exists():
        print(f"Błąd: Nie znaleziono pliku {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(results_data):
    total = len(results_data)
    if total == 0:
        return {}

    top1 = sum(1 for r in results_data if r["correct_result_position"] == 1)
    top3 = sum(1 for r in results_data if r["correct_result_position"] and r["correct_result_position"] <= 3)
    
    positions = [r["correct_result_position"] for r in results_data if r["correct_result_position"] is not None]
    avg_rank = sum(positions) / len(positions) if positions else 0

    avg_embed_time = sum(r["embedding_generation_time_ms"] for r in results_data) / total
    avg_search_time = sum(r["search_time_ms"] for r in results_data) / total

    return {
        "total_queries": total,
        "top1_acc": (top1 / total) * 100,
        "top3_acc": (top3 / total) * 100,
        "avg_rank": avg_rank,
        "avg_embed_time_ms": avg_embed_time,
        "avg_search_time_ms": avg_search_time
    }


def main():
    fixres_data = load_json(FIXRES_PATH)
    naflex_data = load_json(NAFLEX_PATH)

    if not fixres_data or not naflex_data:
        return

    m_fix = calculate_metrics(fixres_data)
    m_flex = calculate_metrics(naflex_data)

    print("\n" + "=" * 65)
    print(" PORÓWNANIE MODELI: SigLIP 2 FixRes vs SigLIP 2 NaFlex")
    print("=" * 65)
    print(f"{'Metryka':<30} | {'FixRes (224px)':<14} | {'NaFlex (Native)':<14}")
    print("-" * 65)
    print(f"{'Przetworzone zapytania':<30} | {m_fix['total_queries']:<14} | {m_flex['total_queries']:<14}")
    print(f"{'Top-1 Accuracy (Dokładność)':<30} | {m_fix['top1_acc']:.1f}%{'':<9} | {m_flex['top1_acc']:.1f}%")
    print(f"{'Top-3 Accuracy':<30} | {m_fix['top3_acc']:.1f}%{'':<9} | {m_flex['top3_acc']:.1f}%")
    print(f"{'Średnia pozycja (im mniej tym lepiej)':<30} | {m_fix['avg_rank']:.2f}{'':<10} | {m_flex['avg_rank']:.2f}")
    print(f"{'Śr. czas generowania tekstu (ms)':<30} | {m_fix['avg_embed_time_ms']:.2f} ms{'':<6} | {m_flex['avg_embed_time_ms']:.2f} ms")
    print(f"{'Śr. czas przeszukiwania bazy (ms)':<30} | {m_fix['avg_search_time_ms']:.2f} ms{'':<6} | {m_flex['avg_search_time_ms']:.2f} ms")
    print("=" * 65)

    # Szukanie zapytań, w których NaFlex okazał się wyraźnie lepszy od FixRes
    print("\nPRZYKŁADY, GDZIE NaFlex WYGRAŁ Z FixRes:")
    improvements = 0
    for fix_q, flex_q in zip(fixres_data, naflex_data):
        pos_fix = fix_q["correct_result_position"]
        pos_flex = flex_q["correct_result_position"]

        if pos_flex is not None and pos_fix is not None and pos_flex < pos_fix:
            improvements += 1
            if improvements <= 5: # Pokazujemy max 5 przykładow
                print(f"  • Plik: {flex_q['image_id']}")
                print(f"    Zapytanie: '{flex_q['query']}'")
                print(f"    Pozycja w FixRes: {pos_fix} ➔ Pozycja w NaFlex: {pos_flex} ✅\n")

    if improvements == 0:
        print("  Brak zapytań, w których NaFlex miał lepszą pozycję niż FixRes.")


if __name__ == "__main__":
    main()