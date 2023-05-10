import socket
import json
from game_state import GameState
#from bot import fight
import model
import sys
import csv
import os
from bot import Bot
def connect(port):
    #For making a connection with the game
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    (client_socket, _) = server_socket.accept()
    print ("Connected to game!")
    return client_socket

def send(client_socket, command):
    #This function will send your updated command to Bizhawk so that game reacts according to your command.
    command_dict = command.object_to_dict()
    pay_load = json.dumps(command_dict).encode()
    client_socket.sendall(pay_load)

def receive(client_socket):
    #receive the game state and return game state
    pay_load = client_socket.recv(4096)
    input_dict = json.loads(pay_load.decode())
    
    game_state = GameState(input_dict)

    return game_state

def write_state(row):
    column_names = [
        'p1_player_id', 'p1_health', 'p1_is_crouching', 'p1_is_jumping', 'p1_is_player_in_move', 'p1_move_id', 'p1_x_coord', 'p1_y_coord',
        'p1_up', 'p1_down', 'p1_right', 'p1_left', 'p1_select', 'p1_start', 'p1_Y', 'p1_B', 'p1_X', 'p1_A', 'p1_L', 'p1_R',
        'p2_player_id', 'p2_health', 'p2_is_crouching', 'p2_is_jumping', 'p2_is_player_in_move', 'p2_move_id', 'p2_x_coord', 'p2_y_coord',
        'p2_up', 'p2_down', 'p2_right', 'p2_left', 'p2_select', 'p2_start', 'p2_Y', 'p2_B', 'p2_X', 'p2_A', 'p2_L', 'p2_R',
        'timer', 'fight_result', 'has_round_started', 'is_round_over'
    ]

    dataset_file = 'dataset.csv'

    if not os.path.exists(dataset_file) or os.path.getsize(dataset_file) == 0:
        with open(dataset_file, 'w', newline='') as f_object:
            print("Writing column names to CSV")
            writer_object = csv.writer(f_object)
            writer_object.writerow(column_names)
            f_object.close()

    with open(dataset_file, 'a', newline='') as f_object:
        print("Writing to CSV")
        writer_object = csv.writer(f_object)
        writer_object.writerow(row)
        f_object.close()

    return

def main():
    if (sys.argv[1]=='1'):
        client_socket = connect(9999)
    elif (sys.argv[1]=='2'):
        client_socket = connect(10000)
    else:
        print("Invalid args")
        return
    current_game_state = None
    ml_model, scaler = model.load_model("trained_model.pkl")

    #print( current_game_state.is_round_over )
    bot = Bot()
    data_list = []
    while (current_game_state is None) or (not current_game_state.is_round_over):

        current_game_state = receive(client_socket)
        data_list.append(current_game_state.get_game_data())
        bot_command = bot.fight(current_game_state, sys.argv[1], ml_model, scaler)
        send(client_socket, bot_command)

    last_row = data_list[len(data_list) - 1]
    if (last_row[len(last_row) - 3] == "P1"):
        for data in data_list:
            write_state(data)
            

if __name__ == '__main__':
   main()
