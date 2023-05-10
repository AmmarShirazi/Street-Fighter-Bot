from player import Player

class GameState:

    def __init__(self, input_dict):
        self.data_dict = input_dict
        self.dict_to_object(input_dict)

    def dict_to_object(self, input_dict):

        self.player1 = Player(input_dict['p1'])
        self.player2 = Player(input_dict['p2'])
        self.timer = input_dict['timer']
        self.fight_result = input_dict['result']
        self.has_round_started = input_dict['round_started']
        self.is_round_over = input_dict['round_over']
        
    def get_game_data(self):

        data_list = []

        p1_list = [self.player1.player_id ,self.player1.health, self.player1.is_crouching, self.player1.is_jumping, self.player1.is_player_in_move, self.player1.move_id, self.player1.x_coord, self.player1.y_coord]
        p1_buttons_list = [self.player1.player_buttons.up, self.player1.player_buttons.down, self.player1.player_buttons.right, self.player1.player_buttons.left, self.player1.player_buttons.select, self.player1.player_buttons.start, self.player1.player_buttons.Y, self.player1.player_buttons.B, self.player1.player_buttons.X, self.player1.player_buttons.A, self.player1.player_buttons.L, self.player1.player_buttons.R]

        p2_list = [self.player2.player_id ,self.player2.health, self.player2.is_crouching, self.player2.is_jumping, self.player2.is_player_in_move, self.player2.move_id, self.player2.x_coord, self.player2.y_coord]
        p2_buttons_list = [self.player2.player_buttons.up, self.player2.player_buttons.down, self.player2.player_buttons.right, self.player2.player_buttons.left, self.player2.player_buttons.select, self.player2.player_buttons.start, self.player2.player_buttons.Y, self.player2.player_buttons.B, self.player2.player_buttons.X, self.player2.player_buttons.A, self.player2.player_buttons.L, self.player2.player_buttons.R]

        misc_data = [self.timer, self.fight_result, self.has_round_started, self.is_round_over]
        
        data_list.extend(p1_list)
        data_list.extend(p1_buttons_list)
        data_list.extend(p2_list)
        data_list.extend(p2_buttons_list)
        data_list.extend(misc_data)

        return data_list
    