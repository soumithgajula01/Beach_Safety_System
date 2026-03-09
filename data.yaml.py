from pathlib import Path

yaml_text = """
train: /home/user/Desktop/Soumith/project/train/images
val: /home/user/Desktop/Soumith/project/valid/images
test: /home/user/Desktop/Soumith/project/test/images

nc: 2
names:
  - drowning_person
  - person
"""

path = Path("/home/user/Desktop/Soumith/project/data.yaml")
path.write_text(yaml_text.strip())

print("✅ data.yaml created:", path)
