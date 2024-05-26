# Annoy Store

## 实现细节

由于不支持存储后再添加或删除，并且只能存储 id 与向量，不能存储额外的文字信息，所以拿 sqlite 持久化数据，每次更改时，先在 sqlite中做对应更改，再用里面的数据重建 annoy 索引。

id 使用 sqlite 的自增 id，保证和 annoy 索引的 id 一致，方便查找对应的文本。
