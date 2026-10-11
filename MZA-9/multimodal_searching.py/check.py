import sys
import traceback
from pathlib import Path
from component import EmbeddingComponent
import docling_szymon


def test_single_pdf(pdf_path_str: str):
    pdf_path = Path(pdf_path_str)
    
    if not pdf_path.exists():
        print(f" BŁĄD: Plik '{pdf_path}' nie istnieje w podanej ścieżce!")
        return

    print(f"\n Rozpoczynam test przetwarzania: {pdf_path.name}")
    print("-" * 60)

    try:
        embedder = EmbeddingComponent(allow_cloud=False)
        print(" Model OpenCLIP załadowany poprawnie.")

        doc, chunks = docling_szymon.process_document(
            input_path=pdf_path,
            ocr_mode="auto",
            embedder=embedder
        )

        print(f"\n SUCCESS: Docling przetworzył plik!")
        print(f" Liczba wygenerowanych chunków: {len(chunks)}")

        for idx, chunk in enumerate(chunks, start=1):
            text_snippet = chunk.get("ocr_text", "").strip().replace("\n", " ")[:60]
            print(f"   [{idx}] Strona {chunk.get('page_number')}: \"{text_snippet}...\"")

        if len(chunks) == 0:
            print("\n UWAGA: Docling zwrócił 0 chunków. Plik nie zostanie zapisany w Qdrancie!")

    except Exception as exc:
        print("\n WYJĄTEK / BŁĄD:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)


if __name__ == "__main__":
    # Zmień na dokładną ścieżkę do Twojego pliku PDF
    target_pdf = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\mateu\Desktop\Zbior_danych\pdf_files\file_4.pdf"
    test_single_pdf(target_pdf)