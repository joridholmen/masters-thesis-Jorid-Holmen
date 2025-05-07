txt_file = 'ute6'
text = f'realtime-res/{txt_file}.txt'
res_file = f'interpret-robot/{txt_file}_res.txt'
unique_ids = {}
track_data = {}


with open(text, "r") as file:
    for line in file:
        if "ID:" in line and "frame:" in line and "Label: " in line and "Coordinates: " in line:
            parts = line.strip().split(", ")
            frame_part = parts[0]
            id_part = parts[1]
            label_part = [p for p in parts if p.startswith("Label:")][0]
            coordinates_parts = [p for p in parts if p.startswith("Coordinates:") or p.startswith("Longitude:")]

            frame = int(frame_part.split(":")[1])
            try:
                obj_id = float(id_part.split(":")[1])
            except:
                obj_id = None
            label = label_part.split(":")[1].strip()

            confidence_part = [p for p in parts if p.startswith("Confidence:")][0]
            confidence = float(confidence_part.split(":")[1].strip())

            # Coordinates may be split across two parts, reconstruct if needed
            coordinates = ", ".join(coordinates_parts).replace("Coordinates: ", "").strip()

            if obj_id is not None:
                if obj_id not in track_data:
                    track_data[obj_id] = {
                        "frames": [],
                        "confidences": [],
                        "label": label,
                        "coordinates": coordinates
                    }
                track_data[obj_id]["frames"].append(frame)
                track_data[obj_id]["confidences"].append(confidence)


# Write average confidence results
with open(res_file, "w") as file:
    for obj_id, info in track_data.items():
        avg_conf = sum(info["confidences"]) / len(info["confidences"])
        first_frame = min(info["frames"])
        file.write(
            f"ID: {obj_id:.0f}, First Frame: {first_frame}, Label: {info['label']}, "
            f"Average Confidence: {avg_conf:.3f}, Coordinates: {info['coordinates']}\n"
        )