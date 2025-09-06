from lxml import etree

def local_name(tag):
    return etree.QName(tag).localname if tag else ""

xml_path = "rawdata/full database.xml"
context = etree.iterparse(xml_path, events=("start", "end"), recover=True, huge_tree=True)

in_drug = False
drug = {}
count = 0

for event, elem in context:
    ln = local_name(elem.tag).lower()

    if event == "start" and ln == "drug":
        in_drug = True
        drug = {"id": None, "name": None}

    elif event == "end" and ln == "drug":
        if drug["id"]:
            print(f"DEBUG DRUG: {drug}")
            count += 1
            if count >= 5:
                break
        in_drug = False
        elem.clear()

    elif in_drug:
        if ln == "drugbank-id" and elem.get("primary") == "true":
            drug["id"] = "".join(elem.itertext()).strip()
        elif ln == "name" and not drug["name"]:
            drug["name"] = "".join(elem.itertext()).strip()
