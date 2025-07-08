from datasets import load_dataset

def get_property_from_data(data, property: str):
    desired_keys = ["text"]
    desired_keys.append(property)
    filtered_data = [
    {key: entry[key] for key in desired_keys}
    for entry in data
    ]
    return filtered_data


train_dataset = load_dataset("google/civil_comments", split="train")
valid_dataset = load_dataset("google/civil_comments", split="validation")
test_dataset  = load_dataset("google/civil_comments", split="test")

#train_dataset = get_property_from_data(train_dataset, "sexual_explicit")
#valid_dataset = get_property_from_data(valid_dataset, "sexual_explicit")
test_dataset  = get_property_from_data(test_dataset, "sexual_explicit")

print(test_dataset[0])

