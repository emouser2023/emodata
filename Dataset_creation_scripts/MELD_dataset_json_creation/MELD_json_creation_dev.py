import csv
import json
import os
from tqdm import tqdm

# Path to your CSV file
csv_file = '/media/sdb_access/1_emotionCLIP/MELD_dataset/dev_sent_emo.csv'
input_json_path ='/media/sdb_access/1_emotionCLIP/MELD_dataset/human+face_acm/dev_bounding_boxes.json'
base_jason_path = '/media/sdb_access/1_emotionCLIP/MELD_dataset/json/dev'
dataset_base_path = '/media/sdb_access/1_emotionCLIP/MELD_dataset/Dataset/'

try:
    with open(input_json_path, "r") as f:
        bbox_data = json.load(f)
except Exception as e:
    print(f"⚠️ Error loading JSON {input_json_path}: {e}")


os.makedirs(base_jason_path, exist_ok=True)

# Emotion to class ID mapping
EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
emotion_to_id = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}

# List to hold dictionaries
utterance_data = []
with open(csv_file, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        emotion = row.get("Emotion", "").strip().lower()
        class_id = emotion_to_id.get(emotion, -1)  # -1 if emotion not found

        data = {
            "Emotion": emotion,
            "class id": class_id,
            "Dialogue_ID": row.get("Dialogue_ID", "").strip(),
            "Utterance_ID": row.get("Utterance_ID", "").strip(),
            "Utterance": row.get("Utterance", "").strip()
        }
        utterance_data.append(data)

# You can print or save the list as needed
# print(utterance_data[:3])  # Print first 3 entries for quick check

for row in tqdm(utterance_data):
    video_name = "dia{}_utt{}.mp4".format(row['Dialogue_ID'],row['Utterance_ID'])
    audio_name = "dia{}_utt{}.mp3".format(row['Dialogue_ID'],row['Utterance_ID'])
    video_path = "/videos/dev_splits/"+video_name
    # audio_path = "/audio/dev_splits/"+audio_name
    bbox_all = bbox_data[video_name]
    speaking_face_boxes = []
    speaking_human_boxes = []
    for bbox_frame  in bbox_all.values():
        # Detect faces and humans using YOLOv8
        face_boxes = [[box['x1'], box['y1'], box['x2'], box['y2']] for box in bbox_frame['faces']]
        human_boxes = [[box['x1'], box['y1'], box['x2'], box['y2']] for box in bbox_frame['humans']]
        speaking_face_boxes.append(face_boxes)
        speaking_human_boxes.append(human_boxes)

    # if not (os.path.exists(os.path.join(dataset_base_path, video_path.lstrip("/"))) and os.path.exists(os.path.join(dataset_base_path,audio_path.lstrip("/")))):
    if not os.path.exists(os.path.join(dataset_base_path, video_path.lstrip("/"))):
        print('video file not available')
        continue

    
    clip_json_name = "dia{}_utt{}.json".format(row['Dialogue_ID'],row['Utterance_ID'])
    clip_data = {
            "video_name":video_name,
            "audio_name":audio_name,
            "video_path": video_path,
            # "audio_path": audio_path,
            "text": row['Utterance'],
            "Emotion": row['Emotion'],
            "class id": row['class id'],
            "face_boxes": speaking_face_boxes,
            "human_boxes":  speaking_human_boxes
        }
    clip_json_path= os.path.join(base_jason_path, clip_json_name)
    with open(clip_json_path, "w") as f:
        json.dump(clip_data, f, indent=4)



