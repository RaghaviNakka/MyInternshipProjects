import nltk
import random
from nltk.chat.util import Chat, reflections

# Download necessary NLTK data
nltk.download('punkt')

# Define some patterns and responses
patterns = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you', ["I'm doing well, thanks!", "I'm good, how about you?"]),
    (r'what is your name', ['My name is MyBot.', "I'm MyBot, nice to meet you!"]),
    (r'bye|goodbye', ['Goodbye!', 'See you later!', 'Bye!']),
    (r'(.*)', ["I'm not sure I understand. Could you rephrase that?", 'Interesting. Tell me more.'])
]

# Create a Chat object
chatbot = Chat(patterns, reflections)

def get_bot_response(user_input):
    return chatbot.respond(user_input)

# Main chat loop
print("Hello! I'm MyBot. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("MyBot: Goodbye!")
        break
    response = get_bot_response(user_input)
    print("MyBot:", response)