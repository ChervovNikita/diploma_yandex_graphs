# Metadata preflight failure

The driver does not support `nvidia-smi --query-gpu=minor_number`. The direct query returned exit 2 and identified that field as invalid. The v2 source reaches that unsupported query after checking the dedicated Python prefix. Three failed metadata launch receipts are preserved. These launches performed no model or dataset work. The v3 source verifies the single allowed UUID before reading its minor from NVIDIA XML metadata and prints structured failure information.
