import requests

url = "https://www.gutenberg.org/files/11/11-0.txt"
response = requests.get(url)

# Be sure to use 'utf-8' encoding to handle special characters correctly
with open('alice_in_wonderland.txt', 'w', encoding='utf-8') as file:
    file.write(response.text)
