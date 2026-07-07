from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

knowledge = open(
    "knowledge.txt",
    encoding="utf-8"
).read()

chunks = knowledge.split("---")

data = []

for chunk in chunks:

    embedding = model.encode(
        chunk
    ).tolist()

    data.append({
        "text": chunk,
        "embedding": embedding
    })

with open(
    "kb.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
        f,
        ensure_ascii=False
    )

print("知识库构建完成")