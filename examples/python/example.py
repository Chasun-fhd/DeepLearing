# import os
#
# from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
# from haystack import Document
#
#
# os.environ["ELASTIC_CLIENT_DISABLE_PRODUCT_CHECK"] = "true"
#
# doc_store = ElasticsearchDocumentStore(hosts="http://localhost:9200", verify_certs=False, timeout=300)
#
# from datasets import load_dataset
#
# subjqa_dataset = load_dataset('subjqa', name="electronics")
#
# import pandas as pd
#
# dfs = {split: dset.to_pandas() for split, dset in subjqa_dataset.flatten().items()}
# for split, df in dfs.items():
#     docs = [Document(content=row["context"], meta={"item_id": row["title"], "question_id": row["id"], "split": split})
#             for _, row in df.drop_duplicates(subset="context").iterrows()]
#     doc_store.write_documents(docs)
#     break
# print(f"Loaded {doc_store.count_documents()} documents")

def echo(*args, **kwargs):
    print(args)
    print(kwargs.get("name1"))

echo("lisi","ddd", name1="bb", cc="dd")