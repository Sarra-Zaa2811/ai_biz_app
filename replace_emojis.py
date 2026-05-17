file_path = r'C:\Users\Msi\Desktop\ai_biz_app\app.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Define replacements
replacements = [
    ('page_icon="⚙"', 'page_icon=""'),
    ('📂 Dataset', 'Dataset'),
    ('🧠 AI BIZ INTEL', 'AI Business Intelligence Platform'),
    ('🔗 Quick Links', 'Quick Links'),
    ('📊 MLflow', 'MLflow'),
    ('🎯 Automated', 'Automated'),
    ('🤖 Generative', 'Generative'),
    ('📊 Interactive', 'Interactive'),
    ('🚀 Cloud', 'Cloud'),
    ('📊 Open MLflow UI ↗', 'Open MLflow UI'),
    ('🚀 Train', 'Train'),
    ('✅ Training', 'Training'),
    ('⚡ Predict', 'Predict'),
    ('✅ Negative', 'Negative'),
    ('⚠️ Positive', 'Positive'),
    ('🤖 Gemini', 'Gemini'),
    ('⚡ Run Batch', 'Run Batch'),
    ('📊 Automated', 'Automated'),
    ('🔍 Generate', 'Generate'),
    ('🧑', '[USER]'),
    ('🤖', '[AI]'),
    ('📋 Action', 'Action'),
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

print(total_replacements)
