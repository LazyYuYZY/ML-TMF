import json

with open("/root/wjd/bf_cm-sketch/pkt/202401031400.json", "r") as file:
    data = json.load(file)

unique_data = []
seen = set()
for item in data:
    flow_tuple = (item["SrcIP"], item["SrcPort"], item["DstIP"], item["DstPort"], item["Protocol"])
    if flow_tuple not in seen:
        unique_data.append(item)
        seen.add(flow_tuple)

with open("/root/wjd/bf_cm-sketch/pkt/dat-test.json", "w") as file:
    json.dump(unique_data, file)
    file.write("\n")