from command import Command
import numpy as np
from buttons import Buttons
from game_state import GameState
import pandas as pd

import model
class Bot:

    def __init__(self):
        #< - v + < - v - v + > - > + Y
        self.fire_code=["<","!<","v+<","!v+!<","v","!v","v+>","!v+!>",">+Y","!>+!Y"]
        self.exe_code = 0
        self.start_fire=True
        self.remaining_code=[]
        self.my_command = Command()
        self.buttn = Buttons()

    def fight(self, current_game_state, player, ml_model, scaler):

        column_names = [
            'p1_player_id', 'p1_health', 'p1_is_crouching', 'p1_is_jumping', 'p1_is_player_in_move', 'p1_move_id', 'p1_x_coord', 'p1_y_coord',
            'p1_up', 'p1_down', 'p1_right', 'p1_left', 'p1_select', 'p1_start', 'p1_Y', 'p1_B', 'p1_X', 'p1_A', 'p1_L', 'p1_R',
            'p2_player_id', 'p2_health', 'p2_is_crouching', 'p2_is_jumping', 'p2_is_player_in_move', 'p2_move_id', 'p2_x_coord', 'p2_y_coord',
            'p2_up', 'p2_down', 'p2_right', 'p2_left', 'p2_select', 'p2_start', 'p2_Y', 'p2_B', 'p2_X', 'p2_A', 'p2_L', 'p2_R',
            'timer', 'fight_result', 'has_round_started', 'is_round_over'
        ]
        data = current_game_state.get_game_data()

        data_dict = dict(zip(column_names, data))
        df = pd.DataFrame([data_dict])
        prediction = model.make_prediction(ml_model, scaler, df)



        #print(prediction)

        prediction_row = prediction[0]
        # ["p1_up", "p1_down", "p1_right", "p1_left", "p1_Y", "p1_B", "p1_X", "p1_A", "p1_L", "p1_R"]
        self.buttn.up = bool(prediction_row[0])
        self.buttn.down = bool(prediction_row[1])
        self.buttn.right = bool(prediction_row[2])
        self.buttn.left = bool(prediction_row[3])
        self.buttn.Y = bool(prediction_row[4])
        self.buttn.B = bool(prediction_row[5])
        self.buttn.X = bool(prediction_row[6])
        self.buttn.A = bool(prediction_row[7])
        self.buttn.L = bool(prediction_row[8])
        self.buttn.R = bool(prediction_row[9])


        self.my_command.player_buttons = self.buttn

        return self.my_command


        # # #python Videos\gamebot-competition-master\PythonAPI\controller.py 1
        # if player=="1":
        #     #print("1")
        #     #v - < + v - < + B spinning

        #     if( self.exe_code!=0  ):
        #        self.run_command([],current_game_state.player1)
        #     diff=current_game_state.player2.x_coord - current_game_state.player1.x_coord
        #     if (diff > 40) :
        #         toss=np.random.randint(3)
        #         if (toss==0):
        #             #self.run_command([">+^+Y",">+^+Y",">+^+Y","!>+!^+!Y"],current_game_state.player1)
        #             self.run_command([">","-","!>","v+>","-","!v+!>","v","-","!v","v+<","-","!v+!<","<+Y","-","!<+!Y"],current_game_state.player1)
        #         elif ( toss==1 ):
        #             self.run_command([">+^+B",">+^+B","!>+!^+!B"],current_game_state.player1)
        #         else: #fire
        #             self.run_command(["<","-","!<","v+<","-","!v+!<","v","-","!v","v+>","-","!v+!>",">+Y","-","!>+!Y"],current_game_state.player1)
        #     elif (diff < -40) :
        #         toss=np.random.randint(3)
        #         if (toss==0):#spinning
        #             #self.run_command(["<+^+Y","<+^+Y","<+^+Y","!<+!^+!Y"],current_game_state.player1)
        #             self.run_command(["<","-","!<","v+<","-","!v+!<","v","-","!v","v+>","-","!v+!>",">+Y","-","!>+!Y"],current_game_state.player1)
        #         elif ( toss==1):#
        #             self.run_command(["<+^+B","<+^+B","!<+!^+!B"],current_game_state.player1)
        #         else: #fire
        #             self.run_command([">","-","!>","v+>","-","!v+!>","v","-","!v","v+<","-","!v+!<","<+Y","-","!<+!Y"],current_game_state.player1)
        #     else:
        #         toss=np.random.randint(2)  # anyFightActionIsTrue(current_game_state.player2.player_buttons)
        #         if ( toss>=1 ):
        #             if (diff>0):
        #                 self.run_command(["<","<","!<"],current_game_state.player1)
        #             else:
        #                 self.run_command([">",">","!>"],current_game_state.player1)
        #         else:
        #             self.run_command(["v+R","v+R","v+R","!v+!R"],current_game_state.player1)
        #     self.my_command.player_buttons=self.buttn

        # elif player=="2":

        #     if( self.exe_code!=0  ):
        #        self.run_command([],current_game_state.player2)
        #     diff=current_game_state.player1.x_coord - current_game_state.player2.x_coord
        #     if (  diff > 60 ) :
        #         toss=np.random.randint(3)
        #         if (toss==0):
        #             #self.run_command([">+^+Y",">+^+Y",">+^+Y","!>+!^+!Y"],current_game_state.player2)
        #             self.run_command([">","-","!>","v+>","-","!v+!>","v","-","!v","v+<","-","!v+!<","<+Y","-","!<+!Y"],current_game_state.player2)
        #         elif ( toss==1 ):
        #             self.run_command([">+^+B",">+^+B","!>+!^+!B"],current_game_state.player2)
        #         else:
        #             self.run_command(["<","-","!<","v+<","-","!v+!<","v","-","!v","v+>","-","!v+!>",">+Y","-","!>+!Y"],current_game_state.player2)
        #     elif ( diff < -60 ) :
        #         toss=np.random.randint(3)
        #         if (toss==0):
        #             #self.run_command(["<+^+Y","<+^+Y","<+^+Y","!<+!^+!Y"],current_game_state.player2)
        #             self.run_command(["<","-","!<","v+<","-","!v+!<","v","-","!v","v+>","-","!v+!>",">+Y","-","!>+!Y"],current_game_state.player2)
        #         elif ( toss==1):
        #             self.run_command(["<+^+B","<+^+B","!<+!^+!B"],current_game_state.player2)
        #         else:
        #             self.run_command([">","-","!>","v+>","-","!v+!>","v","-","!v","v+<","-","!v+!<","<+Y","-","!<+!Y"],current_game_state.player2)
        #     else:
        #         toss=np.random.randint(2)  # anyFightActionIsTrue(current_game_state.player2.player_buttons)
        #         if ( toss>=1 ):
        #             if (diff<0):
        #                 self.run_command(["<","<","!<"],current_game_state.player2)
        #             else:
        #                 self.run_command([">",">","!>"],current_game_state.player2)
        #         else:
        #             self.run_command(["v+R","v+R","v+R","!v+!R"],current_game_state.player2)
        #     self.my_command.player2_buttons=self.buttn
        # return self.my_command



    def run_command( self , com , player   ):

        if self.exe_code-1==len(self.fire_code):
            self.exe_code=0
            self.start_fire=False
            print ("compelete")
            #exit()
            # print ( "left:",player.player_buttons.left )
            # print ( "right:",player.player_buttons.right )
            # print ( "up:",player.player_buttons.up )
            # print ( "down:",player.player_buttons.down )
            # print ( "Y:",player.player_buttons.Y )

        elif len(self.remaining_code)==0 :

            self.fire_code=com
            #self.my_command=Command()
            self.exe_code+=1

            self.remaining_code=self.fire_code[0:]

        else:
            self.exe_code+=1
            if self.remaining_code[0]=="v+<":
                self.buttn.down=True
                self.buttn.left=True
                print("v+<")
            elif self.remaining_code[0]=="!v+!<":
                self.buttn.down=False
                self.buttn.left=False
                print("!v+!<")
            elif self.remaining_code[0]=="v+>":
                self.buttn.down=True
                self.buttn.right=True
                print("v+>")
            elif self.remaining_code[0]=="!v+!>":
                self.buttn.down=False
                self.buttn.right=False
                print("!v+!>")

            elif self.remaining_code[0]==">+Y":
                self.buttn.Y= True #not (player.player_buttons.Y)
                self.buttn.right=True
                print(">+Y")
            elif self.remaining_code[0]=="!>+!Y":
                self.buttn.Y= False #not (player.player_buttons.Y)
                self.buttn.right=False
                print("!>+!Y")

            elif self.remaining_code[0]=="<+Y":
                self.buttn.Y= True #not (player.player_buttons.Y)
                self.buttn.left=True
                print("<+Y")
            elif self.remaining_code[0]=="!<+!Y":
                self.buttn.Y= False #not (player.player_buttons.Y)
                self.buttn.left=False
                print("!<+!Y")

            elif self.remaining_code[0]== ">+^+L" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.L= not (player.player_buttons.L)
                print(">+^+L")
            elif self.remaining_code[0]== "!>+!^+!L" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.L= False #not (player.player_buttons.L)
                print("!>+!^+!L")

            elif self.remaining_code[0]== ">+^+Y" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.Y= not (player.player_buttons.Y)
                print(">+^+Y")
            elif self.remaining_code[0]== "!>+!^+!Y" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.Y= False #not (player.player_buttons.L)
                print("!>+!^+!Y")


            elif self.remaining_code[0]== ">+^+R" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.R= not (player.player_buttons.R)
                print(">+^+R")
            elif self.remaining_code[0]== "!>+!^+!R" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.R= False #ot (player.player_buttons.R)
                print("!>+!^+!R")

            elif self.remaining_code[0]== ">+^+A" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.A= not (player.player_buttons.A)
                print(">+^+A")
            elif self.remaining_code[0]== "!>+!^+!A" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.A= False #not (player.player_buttons.A)
                print("!>+!^+!A")

            elif self.remaining_code[0]== ">+^+B" :
                self.buttn.right=True
                self.buttn.up=True
                self.buttn.B= not (player.player_buttons.B)
                print(">+^+B")
            elif self.remaining_code[0]== "!>+!^+!B" :
                self.buttn.right=False
                self.buttn.up=False
                self.buttn.B= False #not (player.player_buttons.A)
                print("!>+!^+!B")

            elif self.remaining_code[0]== "<+^+L" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.L= not (player.player_buttons.L)
                print("<+^+L")
            elif self.remaining_code[0]== "!<+!^+!L" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.L= False  #not (player.player_buttons.Y)
                print("!<+!^+!L")

            elif self.remaining_code[0]== "<+^+Y" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.Y= not (player.player_buttons.Y)
                print("<+^+Y")
            elif self.remaining_code[0]== "!<+!^+!Y" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.Y= False  #not (player.player_buttons.Y)
                print("!<+!^+!Y")

            elif self.remaining_code[0]== "<+^+R" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.R= not (player.player_buttons.R)
                print("<+^+R")
            elif self.remaining_code[0]== "!<+!^+!R" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.R= False  #not (player.player_buttons.Y)
                print("!<+!^+!R")

            elif self.remaining_code[0]== "<+^+A" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.A= not (player.player_buttons.A)
                print("<+^+A")
            elif self.remaining_code[0]== "!<+!^+!A" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.A= False  #not (player.player_buttons.Y)
                print("!<+!^+!A")

            elif self.remaining_code[0]== "<+^+B" :
                self.buttn.left=True
                self.buttn.up=True
                self.buttn.B= not (player.player_buttons.B)
                print("<+^+B")
            elif self.remaining_code[0]== "!<+!^+!B" :
                self.buttn.left=False
                self.buttn.up=False
                self.buttn.B= False  #not (player.player_buttons.Y)
                print("!<+!^+!B")

            elif self.remaining_code[0]== "v+R" :
                self.buttn.down=True
                self.buttn.R= not (player.player_buttons.R)
                print("v+R")
            elif self.remaining_code[0]== "!v+!R" :
                self.buttn.down=False
                self.buttn.R= False  #not (player.player_buttons.Y)
                print("!v+!R")

            else:
                if self.remaining_code[0] =="v" :
                    self.buttn.down=True
                    print ( "down" )
                elif self.remaining_code[0] =="!v":
                    self.buttn.down=False
                    print ( "Not down" )
                elif self.remaining_code[0] =="<" :
                    print ( "left" )
                    self.buttn.left=True
                elif self.remaining_code[0] =="!<" :
                    print ( "Not left" )
                    self.buttn.left=False
                elif self.remaining_code[0] ==">" :
                    print ( "right" )
                    self.buttn.right=True
                elif self.remaining_code[0] =="!>" :
                    print ( "Not right" )
                    self.buttn.right=False

                elif self.remaining_code[0] =="^" :
                    print ( "up" )
                    self.buttn.up=True
                elif self.remaining_code[0] =="!^" :
                    print ( "Not up" )
                    self.buttn.up=False
            self.remaining_code=self.remaining_code[1:]
        return
