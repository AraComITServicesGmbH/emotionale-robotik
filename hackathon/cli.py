def print_line():
    print("-----------------------")

def print_emotions(emotions):
    print(f"{len(emotions)} Persons with the following emotions: ")
    print(" ".join(emotions))
    print_line()

def print_connect(address=None):
    if address is None:
        print("Connected")
    else:
        print(f"Connected with: {address}")
    print_line()
    
def print_disconnect(address=None):
    if address is None:
        print("Disconnected")
    else:
        print(f"Disconnected from: {address}")
    print_line()
 