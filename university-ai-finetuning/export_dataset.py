import json
from data.student_data import TRAINING_DATA

# Save TRAINING_DATA list directly into a JSON file
with open("dataset.json", "w", encoding="utf-8") as f:
    json.dump(TRAINING_DATA, f, indent=4)

print(f"Successfully exported {len(TRAINING_DATA)} records to dataset.json!")