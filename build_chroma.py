import chromadb
from sentence_transformers import SentenceTransformer

# Embedding模型
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# 创建数据库
client = chromadb.PersistentClient(
    path="./chroma_db"
)

# 创建集合
collection = client.get_or_create_collection(
    name="knowledge_base"
)

# 读取知识库
with open(
    "knowledge.txt",
    "r",
    encoding="utf-8"
) as f:

    knowledge = f.read()

# 切块
chunks = knowledge.split("---")

# 添加到数据库
for i, chunk in enumerate(chunks):

    embedding = model.encode(
        chunk
    ).tolist()

    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embedding]
    )

print("ChromaDB知识库构建完成")