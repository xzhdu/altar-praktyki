import json
from pathlib import Path

#list of trials with their corresponding JSON file paths
TRIALS = {
    "Try 1": Path(r"22_porównanie_wyszukiwań\Full_benchmark\Wyniki_ewaluacji_Qdrant_Wariant1.json"),
    "Try 2": Path(r"22_porównanie_wyszukiwań\Full_benchmark\Wyniki_ewaluacji_Qdrant_Wariant2.json"),
    "Try 3": Path(r"22_porównanie_wyszukiwań\Full_benchmark\Wyniki_ewaluacji_Qdrant_Wariant3.json"),
}


def load_json(path: Path):
    if not path.exists():
        print(f"Błąd: Nie znaleziono pliku {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Błąd odczytu {path.name}: {e}")
        return None


def calculate_metrics(results_data: list) -> dict:
    total = len(results_data)
    if total == 0:
        return {}

    top1 = sum(1 for r in results_data if r.get("correct_result_position") == 1)
    top3 = sum(1 for r in results_data if r.get("correct_result_position") and r["correct_result_position"] <= 3)
    top5 = sum(1 for r in results_data if r.get("correct_result_position") and r["correct_result_position"] <= 5)

    positions = [r["correct_result_position"] for r in results_data if r.get("correct_result_position") is not None]
    avg_rank = sum(positions) / len(positions) if positions else 0

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

#analysis and comparison of OpenCLIP results from multiple trials
def main():
    metrics_by_trial = {}

    print("=== WCZYTYWANIE I ANALIZA WYNIKÓW OPENCLIP ===")
    for trial_name, path in TRIALS.items():
        data = load_json(path)
        if data:
            metrics_by_trial[trial_name] = calculate_metrics(data)
            print(f"  • Pomyślnie wczytano: {trial_name} ({len(data)} zapytań)")

    if not metrics_by_trial:
        print("Brak danych do wyświetlenia. Sprawdź ścieżki do plików JSON.")
        return

    # Wyświetlanie tabeli porównawczej
    headers = list(metrics_by_trial.keys())
    col_w = 16
    table_w = 36 + (col_w + 3) * len(headers)

    print("\n" + "=" * table_w)
    print(" PORÓWNANIE TESTÓW OPENCLIP")
    print("=" * table_w)

    # Nagłówek tabeli
    header_str = f"{'Metryka':<35} | " + " | ".join(f"{h:^{col_w}}" for h in headers)
    print(header_str)
    print("-" * table_w)

    # Wiersze z danymi
    def print_row(label, key, is_float=True, unit=""):
        vals = []
        for h in headers:
            val = metrics_by_trial[h].get(key, 0)
            if is_float:
                val_str = f"{val:.1f}{unit}" if unit == "%" else f"{val:.2f}{unit}"
            else:
                val_str = str(val)
            vals.append(f"{val_str:^{col_w}}")
        print(f"{label:<35} | " + " | ".join(vals))

    print_row("Liczba przeanalizowanych zapytań", "total_queries", is_float=False)
    print_row("Top-1 Accuracy (Dokładność)", "top1_acc", unit="%")
    print_row("Top-3 Accuracy", "top3_acc", unit="%")
    print_row("Top-5 Accuracy", "top5_acc", unit="%")
    print_row("Średnia pozycja (im mniej tym lepiej)", "avg_rank")
    print_row("Śr. czas generowania tekstu", "avg_embed_time_ms", unit=" ms")
    print_row("Śr. czas przeszukiwania bazy", "avg_search_time_ms", unit=" ms")

    print("=" * table_w)

    # Wyznaczenie zwycięzcy pod kątem Top-1 Accuracy
    best_trial = max(metrics_by_trial.items(), key=lambda x: x[1].get("top1_acc", 0))
    print(f"\nNajlepsza próba: {best_trial[0]} z wynikiem Top-1 = {best_trial[1]['top1_acc']:.1f}%\n")


if __name__ == "__main__":
    main()