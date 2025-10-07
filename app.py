from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0  

class WeightedTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, frequency=1):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.frequency = frequency

    def search_prefix(self, prefix):
        """Return the node where the prefix ends, or None if not found."""
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node

    def autocomplete(self, prefix):
        node = self.search_prefix(prefix)
        if not node:
            return []
        words = self._dfs(node, prefix)
        # Sort suggestions by frequency (higher first)
        return sorted(words, key=lambda x: -x[1])

    def _dfs(self, node, prefix):
        results = []
        if node.is_end_of_word:
            results.append((prefix, node.frequency))
        for char, child_node in node.children.items():
            results.extend(self._dfs(child_node, prefix + char))
        return results


# Initialize the weighted Trie
trie = WeightedTrie()

# Insert common words with high frequencies
common_words = {
    # Greetings
    "hello": 100, "hi": 100, "good morning": 95, "good night": 95, 
    "good evening": 90, "bye": 90, "welcome": 85, "thank you": 100,
    
    # Requests & Politeness
    "please": 100, "help": 95, "sorry": 90, "excuse me": 85, 
    "come here": 80, "wait": 80, "stop": 85, "more": 75, 
    "yes": 95, "no": 95, "ok": 85,

    # Questions
    "who": 90, "what": 90, "where": 90, "when": 90, "how": 90, 
    "why": 85, "can": 80, "do": 80, "is": 80, "are": 80,
    
    # Responses & Acknowledgments
    "I am": 90, "you are": 85, "fine": 85, "okay": 85, 
    "understand": 80, "don't understand": 80, "like": 85, "love": 85,
    
    # Emotions
    "happy": 85, "sad": 85, "angry": 80, "excited": 80, 
    "bored": 80, "scared": 75, "tired": 75, "hungry": 80, 
    "thirsty": 80, "fun": 75, "great": 85, "beautiful": 85,

    # Directions & Instructions
    "left": 75, "right": 75, "up": 75, "down": 75, "go": 80, 
    "come": 80, "sit": 80, "stand": 80, "walk": 75, "run": 75,

    # Objects & Places
    "home": 90, "school": 85, "friend": 90, "family": 85, 
    "water": 80, "food": 80, "book": 80, "phone": 80, 
    "car": 75, "bus": 75,

    # Basic Numbers (Important for Sign Language)
    "one": 80, "two": 80, "three": 80, "four": 80, "five": 80, 
    "six": 75, "seven": 75, "eight": 75, "nine": 75, "ten": 75
}

number_to_word = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "0": "zero", "6_s":"six south", "7_s":"seven south", "8_s":"eight south", "9_s":"nine south"
}
  

for word, freq in common_words.items():
    trie.insert(word, freq)

# Insert other words with lower frequency
import nltk
from nltk.corpus import words
for word in words.words():
    trie.insert(word, 10)  # Default frequency for general words


import pyttsx3
import re

# Initialize the TTS engine for speech
tts_engine = pyttsx3.init()

def speak_word(word):
    #tts_engine.say(word)
    # tts_engine.runAndWait()
    pass


def on_speak(input_text):
    input_text = input_text.lower()
    if input_text[-1] in number_to_word:
        suggestion_word = number_to_word[input_text[-1]]
        speak_word(suggestion_word)
    elif len(input_text)>=3 and input_text[-3:] in number_to_word:
        suggestion_word = number_to_word[input_text[-3:]]
        speak_word(suggestion_word)
    else:
        speak_word(input_text[-1])

        # Find all positions of spaces and digits
        matches = list(re.finditer(r"[ \d]", input_text))

        # Get the last match if there are any
        if matches:
            last_match = matches[-1]  # The last match in the list
            input_text=(input_text[last_match.start()+1:])

        suggestions = trie.autocomplete(input_text)[:3]  

        if suggestions:
            speak_word(suggestions[0][0])

        return suggestions






import cv2
import numpy as np
import pickle
import time
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

with open('d_model.pkl', 'rb') as file:
        model = pickle.load(file)



char_map={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '6_S', 11: '7_S', 12: '8_S', 13: '9_S',14: 'A', 15: 'B', 16: 'C', 17: 'D', 18: 'E', 19: 'F', 20: 'G', 21: 'H', 22: 'I', 23: 'J', 24: 'K', 25: 'L', 26: 'M', 27: 'N', 28: 'O', 29: 'P', 30: 'Q', 31: 'R', 32: 'S', 33: 'T', 34: 'U', 35: 'V', 36: 'W', 37: 'X', 38: 'Y', 39: 'Z', 40:"Hello", 41:"Bye", 42:"Thank You", 43:"Sorry"}

# char_map={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

# char_map={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '6 South', 11: '7 South', 12: '8 South', 13: '9 South',14:"Hello", 15:"Bye", 16:"Thank You", 17:"Sorry"}

sentence = ''
suggestions=[]
cap = cv2.VideoCapture(0) 
current_char = None
last_detected_time = 0
detection_delay = 2 
detected_char = ''
lock = threading.Lock()  # To prevent race conditions in multi-threading
current_prediction = ""
current_suggestions = []
current_sentence = ""

def extract_landmarks_from_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_vector = []
            for lm in hand_landmarks.landmark:
                landmark_vector.extend([lm.x, lm.y, lm.z])
            return np.array(landmark_vector)
    return None



app = Flask(__name__)
camera = cv2.VideoCapture(0)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')



def generate_frames():
    global current_char, last_detected_time, detected_char, sentence, suggestions
    global current_prediction, current_suggestions,current_sentence

    while True:
        try:
            with lock:  
                ret, frame = cap.read()
                if not ret:
                    break

                landmark_vector = extract_landmarks_from_image(frame)

                if landmark_vector is not None:
                    landmark_vector = np.expand_dims(landmark_vector, axis=0)  
                    
                    predicted_class = model.predict(landmark_vector)
                    character = char_map[predicted_class[0]]
                    
                    if character == current_char:
                        if time.time() - last_detected_time >= detection_delay:
                            detected_char = character
                            sentence += detected_char
                            current_char = None  
                            suggestions = on_speak(sentence) 
                            current_prediction = detected_char
                            current_sentence = sentence
                            current_suggestions = [sug[0] for sug in suggestions]                  
                             
                            
                    else:
                        current_char = character
                        last_detected_time = time.time()



                    # Encode the frame in JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame as an HTTP response with content type multipart/x-mixed-replace
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error during frame processing: {e}")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict_char', methods=['POST'])
def predict_char():
    # This function should be replaced with your model's prediction logic
    # Currently, it's a placeholder for demonstration.
    predicted_char = "A"  # Replace with actual prediction
    return jsonify({
        'predicted_char': predicted_char,
        'sentence': "Example sentence",
        'suggestions': ["Example sentence 1", "Example sentence 2", "Example sentence 3"]
    })

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    return jsonify({
        'prediction': current_prediction,
        'suggestions': current_suggestions,
        'sentence' : current_sentence
    })

if __name__ == "__main__":
    app.run(debug=True)
