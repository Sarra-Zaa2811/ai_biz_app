#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Remove emojis from app.py for professional appearance."""

import os

file_path = r'C:\Users\Msi\Desktop\ai_biz_app\app.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define replacements
replacements = [
    ('[🐙 DagsHub Repo]', '[DagsHub Repo]'),
    ('**📦 Datasets**', '**Datasets**'),
    ('🔵 Classification', 'Classification'),
    ('🟡 Regression', 'Regression'),
    ('- 🎯 **Automated ML pipelines**', '- **Automated ML pipelines**'),
    ('- 🏆 **Champion model management**', '- **Champion model management**'),
    ('- 📊 **Interactive dashboards**', '- **Interactive dashboards**'),
    ('- 🚀 **Cloud deployment**', '- **Cloud deployment**'),
    ('⏳ ', ''),
    ('🏆 <b>Champion:</b>', '<b>Champion:</b>'),
    ('📄 ', ''),
    ('💬 Ask Anything About Your Data', 'Ask Anything About Your Data'),
    ('"🏠 Présentation"', '"Home"'),
    ('"🧪 Training & MLflow"', '"Training & MLflow"'),
    ('"⚡ Prédiction Unitaire"', '"Single Prediction"'),
    ('"📦 Prédiction Batch"', '"Batch Prediction"'),
    ('"📊 Dashboard"', '"Dashboard"'),
]

# Count replacements
total_replacements = 0
for old, new in replacements:
    count = content.count(old)
    if count > 0:
        total_replacements += count
        content = content.replace(old, new)

# Write back the file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Total replacements made: {total_replacements}")
