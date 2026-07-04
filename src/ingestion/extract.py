"""PDF partitioning and element extraction using unstructured."""

from unstructured.partition.pdf import partition_pdf

from config.settings import (
    PDF_STRATEGY,
    PDF_EXTRACT_IMAGE_BLOCK_TYPES,
    PDF_INFER_TABLE_STRUCTURE,
)


def partition_document(file_path: str) -> list:
    """Extract elements from a PDF using unstructured's hi-res strategy.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of extracted unstructured elements.
    """
    print(f"Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,
        strategy=PDF_STRATEGY,
        infer_table_structure=PDF_INFER_TABLE_STRUCTURE,
        extract_image_block_types=PDF_EXTRACT_IMAGE_BLOCK_TYPES,
        extract_image_block_to_payload=True,
    )

    print(f"Extracted {len(elements)} elements")
    return elements
