import numpy as np
import json
# 假设这是你的字典，包含了你上面列出的数据
def load_parameters():
    f = open('./embed_500d.vec', "r")
    parameters = json.loads(f.read())
    f.close()
    return parameters
data_dict = load_parameters()

# 获取'ent_embeddings.weight'对应的张量或数组
target_tensor = data_dict["ent_embeddings.weight"]

# 将张量转换为NumPy数组
target_array = np.array(target_tensor)


# 假设你想要找的值是 0.015222427435219288
search_value = -0.00134227246046066

tolerance = 1e-7
indices = np.where(np.isclose(target_array, search_value, atol=tolerance))

# 输出索引
print("Index of the value:", indices)