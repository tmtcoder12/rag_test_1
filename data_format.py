import json

# Load the JSON data
with open("data.json", "r", encoding="utf-8") as f:
    patients = json.load(f)

def visit_to_text(patient, visit):
    """Create a clean, comma-separated text string for a single visit."""
    # Patient-level context
    parts = [
        f"Patient_ID: {patient.get('Patient_ID', '')}",
        f"Name: {patient.get('Name', '')}",
        f"Age: {patient.get('Age', '')}",
        f"Gender: {patient.get('Gender', '')}",
        f"Occupation: {patient.get('Occupation', 'N/A')}",
        f"Allergies: {', '.join(patient.get('Allergies', [])) or 'None'}",
        f"Past_Medical_History: {', '.join(patient.get('Past_Medical_History', [])) or 'None'}",
        f"Family_History: {'; '.join([f'{k}: {v}' for k, v in patient.get('Family_History', {}).items()]) or 'None'}",
        f"Current_Medications: {', '.join(patient.get('Current_Medications', [])) or 'None'}",
        f"Visit_Date: {visit.get('Visit_Date', '')}",
        f"Reason_for_Visit: {visit.get('Reason_for_Visit', '')}",
        f"Notes: {visit.get('Notes', '')}",
        f"Assessment: {visit.get('Assessment', '')}",
        f"Plan: {visit.get('Plan', '')}"
    ]
    
    # Join with commas and remove extra spaces
    text = ", ".join(parts)
    return text.strip()

# Build one record per visit
visit_records = []
for patient in patients:
    for visit in patient.get("Visits", []):
        visit_id = f"{patient['Patient_ID']}_{visit['Visit_Date']}"
        visit_records.append({
            "id": visit_id,
            "patient_id": patient["Patient_ID"],
            "text": visit_to_text(patient, visit)
        })

# Print a few for verification
for rec in visit_records[:2]:
    print(f"\n--- {rec['id']} ---\n{rec['text']}\n")

# Save to JSON file
with open("formatted_data.json", "w", encoding="utf-8") as f:
    json.dump(visit_records, f, ensure_ascii=False, indent=2)
