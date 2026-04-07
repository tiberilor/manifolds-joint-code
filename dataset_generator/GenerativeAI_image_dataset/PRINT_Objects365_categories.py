import pickle
from pprint import pprint

pkl_path = "path/to/Objects365_categories.pkl"  # Pickle file to inspect.

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(f"Number of entries: {len(data)}")

# print first few keys
print("First keys:")
for i, k in enumerate(data.keys()):
    print(k)
    if i == 9:
        break

# print full structure for one example
first_key = next(iter(data))
print("\nExample entry:")
print(first_key)
pprint(data[first_key])

# print category -> detection_names mapping
print("\nCategory to allowed detection labels:")
for key, value in data.items():
    gen = value.get("generation_name")
    det = value.get("detection_names")
    print(f"{key}: generation_name={gen}, detection_names={det}")