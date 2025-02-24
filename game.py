import re
import whisper
import pyaudio
import keyboard
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from image import describe_image

# Parameters for audio recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

WHISPER_MODEL_PATH = "whisper/base.en.pt"  # Local path for Whisper model

class Game:
    def __init__(self):
        self.llm = ChatOllama(
            model="deepseek-r1:14b",
            temperature=".6"
        )
        self.story_context = ""

    def generate_story(self, text, start=False, image_description=""):
        story = ""
        if start:
            story = '''Start the story by describing the player's surroundings and the current situation. 
                    Create a natural introduction to the adventure. Introduce who the main character is, the setting, as well as the sidekick.'''
        messages = [
            HumanMessage(content=
                                "You are a master storyteller, narrating an immersive, novel-like adventure. "
                                "This is not just a game—it is a journey. The player should feel like they are living inside a novel. "
                                "Each scene should be vivid and atmospheric, using rich descriptions to bring the world to life. "
                                "Characters should have depth, motivations, and unique personalities. "
                                "The story should flow naturally, reacting to the player's choices as if unfolding in a book. "
                                "Events should not feel isolated—actions from previous moments should carry consequences into future ones. "
                                "The player’s special powers should gradually evolve, revealing new potential through key moments in the story. "
                                "The player's adventure should involve friendships, rivalries, challenges, and an overarching quest. "
                                "The player will have a speaking animal companion who provides guidance, humor, and insight. They can ask this sidekick for advice. "
                                "The sidekick should have a distinct personality and offer unique perspectives on the story."
                                "Companians of the player should have interesting personalities."
                                "Each response must follow this format:\n\n"
                                
                                "### Scene Description ###\n"
                                "[Write a detailed, immersive continuation of the story based on the player's last action. Use novel-style narration.]\n\n"
                                
                                "### Dialogue ###\n"
                                "[Include dialogue from characters to make the scene feel alive and dynamic.]\n\n"
                                
                                "### Potential Choices ###\n"
                                "- [Provide meaningful choices for the protagonist to shape the story. Each choice should lead to a new, compelling path]\n\n"
                                
                                "Do NOT force the player into a specific decision. "
                                "Wait for their input before continuing, ensuring each choice has weight and impact."
                                "Continue the story based on the player's action and provide the next choices."
                                "If an image description is provided instead of player action, allow the image to influnce the story in some way."
                                f"Story context so far: {self.story_context}\n"
                                f"Player action: {text}\n{story}\n"
                                f"Image description: {image_description}\n\n")
        ]
        response = ""
        skip = True
        for chunk in self.llm.stream(messages):
            response += chunk.content
            if not skip:
                print(chunk.content, end="")
            elif "</think>" not in response and skip:
                continue
            elif "</think>" in response:
                skip = False
                first = re.sub(r"^.*?</think>\s*", "", response, flags=re.DOTALL)
                print(first, end="")
            
        self.story_context += f"\nPlayer input: {text}\n"
        self.story_context += f"\nNarrator: {response}"
        
    def sidekick_response(self, text):
        """Generates a response from the sidekick character based on the player's input."""
        messages = [
                HumanMessage(content=
                                    "You are a master storyteller, narrating an immersive, novel-like adventure. "
                                    "This is not just a game—it is a journey. The player should feel like they are living inside a novel. "
                                    "Each scene should be vivid and atmospheric, using rich descriptions to bring the world to life. "
                                    "Characters should have depth, motivations, and unique personalities. "
                                    "The story should flow naturally, reacting to the player's choices as if unfolding in a book. "
                                    "Events should not feel isolated—actions from previous moments should carry consequences into future ones. "
                                    "The player’s special powers should gradually evolve, revealing new potential through key moments in the story. "
                                    "The player's adventure should involve friendships, rivalries, challenges, and an overarching quest. "
                                    "The player will have a speaking animal companion who provides guidance, humor, and insight. They can ask this sidekick for advice. "
                                    "The sidekick should have a distinct personality and offer unique perspectives on the story."
                                    "Companians of the player should have interesting personalities."
                                    "The player is currently asking the sidekick for advice. Respond matching the personality of the sidekick."
                                    "Each response must follow this format:\n\n"
                                    
                                    "### Dialogue ###\n"
                                    "[Include dialogue from characters to make the scene feel alive and dynamic.]\n\n"
                                    
                                    "### Potential Choices ###\n"
                                    "- [Provide meaningful choices for the protagonist to shape the story. Each choice should lead to a new, compelling path]\n\n"
                                    
                                    "Do NOT force the player into a specific decision. "
                                    "Wait for their input before continuing, ensuring each choice has weight and impact."
                                    "Continue the story based on the player's action and provide the next choices."
                                    f"Story context so far: {self.story_context}\n"
                                    f"Player action: {text}\n\n")
            ]
        response = ""
        skip = True
        for chunk in self.llm.stream(messages):
            response += chunk.content
            if not skip:
                print(chunk.content, end="")
            elif "</think>" not in response and skip:
                continue
            elif "</think>" in response:
                skip = False
                first = re.sub(r"^.*?</think>\s*", "", response, flags=re.DOTALL)
                print(first, end="")
            
        self.story_context += f"\nPlayer input: {text}\n"
        self.story_context += f"\nNarrator: {response}"
    

def record_audio():
    """Records audio while the spacebar is held down and returns the raw audio data as a NumPy array."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    
    print("Hold spacebar to record...")
    while not keyboard.is_pressed('space'):
        pass  # Wait for the spacebar to be pressed
    
    print("Recording...")
    while keyboard.is_pressed('space'):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))
    
    print("Recording stopped.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveform = np.concatenate(frames).astype(np.float32) / 32768.0  # Normalize
    return waveform

def transcribe_audio(audio_data):
    """Loads the Whisper model from a local path and transcribes the recorded audio data."""
    model = whisper.load_model(WHISPER_MODEL_PATH)  # Use local path for Whisper model
    print("Transcribing...")
    result = model.transcribe(audio_data, fp16=False)
    return result["text"]

def use_text_input():
    """Allows the user to choose between text input and voice input."""
    choice = input("Enter 't' for text input or 'v' for voice input: ").strip().lower()
    if choice == 't':
        return True
    elif choice == 'v':
        return False
    else:
        print("Invalid choice. Defaulting to text input.")
        return True
    
def main():
    use_text = use_text_input()
    game = Game()
    print("Welcome to the text-based adventure game!\nType 'help' for instructions.\nType 'exit' to quit.")
    print(game.generate_story("", start=True))
    while True:
        answer = input("Do you want to use an image? (y/n): ").strip().lower()
        if answer == 'y':
            image_path = input("Enter the path to the image: ")
            game.generate_story("", image_description=describe_image(image_path))
            
        print("say/type sidekick to ask for sidekick advice")
        if use_text:
            text = input("Your action: ")
        else:
            audio_data = record_audio()
            text = transcribe_audio(audio_data)
            print(f"Transcribed: {text}")
        
        if text.lower() == 'exit':
            break
        elif text.lower() == 'help':
            print("Type 'exit' to quit the game. Type 'sidekick' to ask for sidekick advice. Enter input to continue the story.")
            continue
        
        if text.lower() == 'sidekick':
            game.sidekick_response(text)
        else:
            game.generate_story(text)
    
    
    
if __name__ == "__main__":
    main()