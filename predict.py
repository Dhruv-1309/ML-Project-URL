"""Load a trained phishing model and predict URL safety."""

from __future__ import annotations

from model_service import predict_url


def main() -> None:
    # Ask user for a URL. If empty, use a simple default example.
    input_url = input("Enter URL to check: ").strip()
    if not input_url:
        input_url = "https://example.com"

    result = predict_url(input_url)

    if result.predicted_class == 1:
        print(f"PHISHING ⚠️  (confidence: {result.confidence:.2%})")
    else:
        print(f"SAFE ✅ (confidence: {result.confidence:.2%})")


if __name__ == "__main__":
    main()
