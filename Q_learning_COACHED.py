from math import gamma
import random
import pickle
from typing import final
import numpy as np

from bot import Agent as coach_agent


DELTA = -0.01
LR = 0.02                  #   "learning rate, High learning rate means faster learning"                     
GAMMA = 0.99                #   "gamma = discount factor. High gamma value means focus on future rewards "
INITIAL_EPSILON = 1.200     #   initial epsilon value

class Agent:
    # REWARD TABLE-------------------------------------------------------------------------------------------------------------------
    DESTROY_SOFTBLOCK           =       30       
    DO_DAMAGE                   =       400
    TAKE_DAMAGE                 =      -100       
    EARN_AMMO                   =       50
    COLLIDE_WITH_WALLS          =       0
    DAMAGE_ITSELF               =       -700        #its own bomb explodes on itself
    NO_DESTRUCTION              =       -5                 #  bomb explodes without destroying any blocks
    NO_KILL                     =       -5              # bomb explodes without killing any player
    PLACING_BOMBS_WITH_NO_AMMO  =       -5
    exist_penalty               =       0
    CLOSENESS_TO_ENEMY          =       0.75         
    #IN_RANGE_OF_BOMB            =       -10
    #Bomb_in_range = 2
    #--------------------------------------------------------------------------------------------------------------------------------
    
                    
    def __init__(self):
        '''
        Place any initialization code for your agent here (if any)
        '''
        self.return_sum = 0                     # this stores sum total reward in every iteration
        self.old_state = None                   #  old state (relevant for learning)
        self.old_action = None                  #  old action (relevant for learning )
        self.episode_number = 0             # sets the episode number
        self.Q_Table = None             #  Q Table
        self.epsilon = None       #  epsilon
        self.setup_episode()
        self.setup_epsilon()
        self.setup_QTable()
        pass

    def learn(self, state, state2, reward, action, action2):
        action = action.index(1)     # convert old action from list([0,0,1,0,0,0]) to index number
        action2 = action2.index(1)   # convert new action from list([0,0,1,0,0,0]) to index number
        # if state already in Q table, then it means that particlar state has been explored, otherwise, new entry is added to the table
        if(state not in self.Q_Table):                     
            self.Q_Table[state] = np.random.uniform(0, 0.5, 6)
        if(state2 not in self.Q_Table):
            self.Q_Table[state2] = np.random.uniform(0, 0.5, 6)
        print(self.Q_Table[state])
        # calculate predict and target
        predict = self.Q_Table[state][action].item()      # valuee for particular state and action given by the state
        target = reward + GAMMA * np.argmax(self.Q_Table[state2]).item()     # optimal value
        #update the q value for the particular state and action 
        self.Q_Table[state][action] = self.Q_Table[state][action].item() + LR * (target - predict) #updating the self.q_value


#--------------------------------------------------------------------------------------------------------------
    #helper function for training_state to identify whether the agent is safe or not
    #   detects possible danger areas marked by bomb explosions
#--------------------------------------------------------------------------------------------------------------------
    #epsilon greedy method to determine which action to choose. 
    #   a random number is generated between 0 and 1. 
    #   if it's smaller than epsilon, then a random move is implemented
    #   else action is taken from the q table

#----------------------------------------------------------------------------------------------------------
    def next_move(self, game_state, player_state):
        '''
        This method is called each time your Agent is required to choose an action
        '''
        # if game tick is 0, there is no old_state or old_action to train the agent, therefore, we just pass this state without training
        # we set value for old_action and old_state from the tick _number state
        if(game_state.tick_number==0):
            current_state = self.get_state_for_agent(game_state, player_state)
            self.old_state = self.get_state_for_agent(game_state, player_state)
            ac = self.get_action(self.old_state, game_state, player_state)
            self.old_action = ac
            ac = ['','u','d','l','r','p'][ac.index(1)]

            return ac
        x_agent = player_state.location[0]  #x coordinate of agent
        y_agent = player_state.location[1]  #y cooordinate of agent
        list_of_enemy = game_state.opponents(player_state.id)
        x_enemy = list_of_enemy[0][0] #x coordinates of agent
        y_enemy = list_of_enemy[0][1] #y coordinates of agent
        #print(self.get_distance_to_enemy([x_agent, y_agent], [x_enemy, y_enemy]))

        # initialize values for training state
        reward = self.get_reward_for_agent(game_state._occurred_event)
        new_state = self.get_state_for_agent(game_state, player_state)
        new_action = self.get_action(new_state, game_state, player_state)                 # action to be implemented in current state

        # just extra reward adjustment for telling the agent not to use p when there are no bombs
        if(player_state.ammo==0 and new_action=="p"):           # check if player still has ammo
            self.reward = self.reward - 10


        reward+=(12-self.manhattan_distance([x_agent, y_agent], [x_enemy, y_enemy]))*self.CLOSENESS_TO_ENEMY
        #print("Here it's is", (12-self.manhattan_distance([x_agent, y_agent], [x_enemy, y_enemy]))*self.CLOSENESS_TO_ENEMY)
        #print(10-self.get_distance_to_enemy(x_enemy,y_enemy))*self.CLOSENESS_TO_ENEMY
        #update return_sum
        self.return_sum+=reward             # update return_sum for calculating cumulative reward

        #learn into q table
        self.learn(state = self.old_state, state2 = new_state, reward = reward, action = self.old_action, action2 = new_action) #learn method to learn from the state
        #print([self.old_state, new_state, reward, self.old_action, new_action])
        #compute if game is over
        done = (game_state.is_over) or (player_state.hp==0) or (game_state.tick_number==1800)
        if(done):
            print("----------------Q_Learning : ------------------------------")
            if(player_state.hp==0):
                print("LOST")
            else:
                print("WIN")
            self.increment_episode()
            self.shift_epsilon(DELTA)
            # store the information from the agent
            self.store_progress()
            self.display_agent_result()     
            self.return_sum = 0

        self.old_action = new_action
        self.old_state = new_state

        #print('Chosen action: ', new_action, '\n')

        new_action = ['','u','d','l','r','p'][new_action.index(1)]

        return new_action


    
    def manhattan_distance(self, start, end):
        distance = abs(start[0]-end[0] + abs(start[1]-end[1]))
        return distance       


    def setup_episode(self):
        '''
        This method loads episode number from file memory if it exists. Otherwise, it creates an instance of the game number and saves it to file
        '''
        try:
            with open('Q_Learning_G_Number', 'rb') as f:
                self.episode_number = pickle.load(f)
        except:
                self.episode_number = 0
                with open("Q_Learning_G_Number","wb") as f:
                    pickle.dump(self.episode_number, f)

    def increment_episode(self):
        '''
        Called when match is over to increase game number
        '''
        self.episode_number+=1

    def get_episode_number(self):
        '''
        GET method for episode number
        '''
        return self.episode_number

    def setup_QTable(self):
        '''
        If the agent already has a QTable, it will use that. Else, it will create an empty q table dictionary.
        '''
        try:
            with open('Q_Learning_Q_TABLE', 'rb') as f:
                self.Q_Table = pickle.load(f)
        except:
                self.Q_Table = dict()
                with open("Q_Learning_Q_TABLE","wb") as f:
                    pickle.dump(self.Q_Table, f)


    def setup_epsilon(self, EPS=None):
        '''
        If agent already has an epsilon value stored in memory, it will use that. Else, it will create new epsilon value, as asked by the user.
        '''
        try:
            with open('Q_Learning_epsilon', 'rb') as f:
                self.epsilon = pickle.load(f)
        except:
            self.epsilon = 1.2
            with open("Q_Learning_epsilon","wb") as f:
                    pickle.dump(self.epsilon, f)


    def shift_epsilon(self, delta):
        '''
        Change epsilon as per the participant's function
        '''
        self.epsilon+=delta

    def set_old_action(self, ac):
        '''
        SET method for old_action
        '''
        self.old_action = ac

    def set_old_state(self, st):
        '''
        SET method for old_state
        '''
        self.old_state = st

    def get_old_action(self):
        '''
        GET method for old_action
        '''
        return self.old_action
    
    def get_old_state(self):
        '''
        GET method for old_state
        '''
        return self.old_state

    def get_distance_to_enemy(self, self_loc, enemy_loc):
        '''
        this method returns a string value of distance. The distance is scaled to make it single digit
        '''
        distance = self.manhattan_distance(self_loc, enemy_loc) 
        return str(int((distance)*0.9))


    def agent_tile_sense(self, x, y):
        '''
        This method returns a list of tile coordinates around the agent that will be visible to the agent.
        Returns a list of 9 coordinates
         ______ ______ ______ ______ ______

        |      |      |  ft  |      |      |
         ------ ------ ------ ------ ------
        |      |  a   |  nt  |  b   |      |
         ------ ------ ------ ------ ------
        |  fl  |  nl  |  own |  nr  |  fr  |
         ------ ------ ------ ------ ------
        |      |  c   |  nb  |  d   |      |
         ------ ------ ------ ------ ------
        |      |      |  fb  |      |      |
         ------ ------ ------ ------ ------
        '''
        a = (x-1,y+1)
        b = (x+1,y+1)
        c = (x-1,y-1)
        d = (x+1,y-1)


        #4 next closes tiles
        fl = (x-2, y)       # 2nd tile left of agent
        ft = (x, y+2)       # 2nd tile top of agent
        fr = (x+2, y)       # 2nd tile right of agent
        fb = (x, y-2)       # 2nd tile bottom agent

        #4 closet tiles
        nl = (x-1,y)        # 1st tile left of agent
        nt = (x,y+1)        # 1st tile top of agent
        nr = (x+1,y)        # 1st tile right of agent
        nb = (x,y-1)        # 1st tile bottom of agent

        #agent's own tiles
        own = (x,y)
        tile_sense = [fl,ft,fr,fb,nl,nt,nr,nb,own,a,b,c,d]
        return tile_sense
        

    def get_state_for_agent(self, game_state, player_state):
        '''
        This method returns a state for the agent. 
        It's a list of length 
        [entity at cell1, 
            ,,     cell2, 
            ,,     cell3,
            ...
            ...
            ,,     cell9],
            distance to enemy,
            contains ammo or not
        '''
        x_agent = player_state.location[0]  #x coordinate of agent
        y_agent = player_state.location[1]  #y cooordinate of agent
        # add visible tiles to the state
        visible_tiles = self.agent_tile_sense(x_agent,y_agent)
        training_state = self.get_tile_state(visible_tiles, game_state, player_state)
        # add enemy distance to the state
        list_of_enemy = game_state.opponents(player_state.id)
        x_enemy = list_of_enemy[0][0] #x coordinates of agent
        y_enemy = list_of_enemy[0][1] #y coordinates of agent
        training_state.append(self.get_distance_to_enemy([x_agent, y_agent], [x_enemy, y_enemy]))
        #add if ammo remaining or not
        if(player_state.ammo==0): training_state.append("0")
        else: training_state.append("1")
        #convert list to a string
        training_state = ''.join(training_state)
        return training_state
        

    def get_tile_state(self, visible_tiles, game_state, player_state):
        '''
        This method queries tiles visible to the agent to understand what object is present there. 
        '''
        training_state = []
        for pt in visible_tiles:
            if(not game_state.is_in_bounds(pt)): training_state.append("0")
            else:
                obs = game_state.entity_at(pt)
                if(obs==None):                  training_state.append("1") # no object foubnd
                elif(obs=="sb"):                training_state.append("2") # breakable wall
                elif(obs=="a"):                 training_state.append("3") # ammo found
                elif(obs=="b"):                 training_state.append("4") # bomb found
                elif(obs==player_state.id):     training_state.append("5") # self detected
                elif(isinstance(obs, int)):     training_state.append("6") # opponent detected
        return training_state

    def get_reward_for_agent(self,occurred_event):
        '''
        This method calculate the reward for a tick of the game. It checks the occurred event
        '''
        #print("Q_Learning\t\t\t",occurred_event)
        reward = 0
        if(occurred_event[0]==1):         reward+=self.EARN_TREASURE
        if(occurred_event[1]> 0):         reward+=(self.DESTROY_SOFTBLOCK*occurred_event[1])
        if(occurred_event[2]> 0):         reward+=(self.DESTROY_ORE*occurred_event[2])
        if(occurred_event[3]==1):         reward+=self.DO_DAMAGE
        if(occurred_event[4]==1):         reward+=self.TAKE_DAMAGE
        if(occurred_event[5]==1):         reward+=self.EARN_AMMO
        if(occurred_event[6]==1):         reward+=self.COLLIDE_WITH_WALLS
        if(occurred_event[7]==1):         reward+=self.DAMAGE_ITSELF   #it's own bomb explodes on itself
        if(occurred_event[8]==1):         reward+=self.NO_DESTRUCTION #  bomb explodes without destroying any blocks
        if(occurred_event[9]==1):         reward+=self.NO_KILL      # bomb explodes without killing any player
        if(occurred_event[10]==1):        reward+=self.PLACING_BOMBS_WITH_NO_AMMO
        reward = reward + self.exist_penalty
        return reward

    def get_action(self, state, game_state, player_state):
        '''
        This method implements epsilon greedy strategy to choose action taken by the agent
        
        a random number is generated between 0 and 1. 
        if it's smaller than epsilon, then a random move is implemented
        else action is taken from the q table
        '''
        final_move = [0,0,0,0,0,0]
        if(self.explore_or_exploit()):
            move = self.explore(game_state,player_state)
        else:
            move = self.exploit(state)
        final_move[move]=1
        return final_move


        
    def explore_or_exploit(self):
        '''
        This method decides whether the agent should explore or not
        TRUE = exploration
        FALSE = exploitation
        '''
        return np.random.uniform(0,1)<1


    def explore(self, game_state, player_state):
        '''
        This method returns a random move. There is only a 1% chance of placing a bomb though

        Ok over here the agent is going to be coached
        '''
        coach_sahab = coach_agent()
        new_action = coach_sahab.next_move(game_state, player_state)

        print('Chosen action: ', new_action, '\n')

        y = ['','u','d','l','r','p'].index(new_action)
        return y

        #x = self.action_space()
        #x = random.random()
        #if(x>0.99):     return  5
        #else:           return random.randint(0, 4)

    def exploit(self, state):
        '''
        This method chooses the best move from the random state
        '''
        if(state not in self.Q_Table): return random.randint(0, 5)
        print(self.Q_Table[state ])
        return np.argmax(self.Q_Table[state])
        

    
    def store_progress(self):
        '''
        This stores agent progress in a file.
        '''
        with open("Q_Learning_Q_TABLE","wb") as Q_table:
                pickle.dump(self.Q_Table, Q_table)
        with open("Q_Learning_G_Number", "wb") as f:
                pickle.dump(self.episode_number, f)
        with open("Q_Learning_epsilon", "wb") as f:
                pickle.dump(self.epsilon, f)

    def display_agent_result(self):
        print("Game Number/t/t:/t", self.episode_number)
        print("Reward Earned/t/t:/t", self.return_sum)
        print("Epsilon/t/t:/t", self.epsilon)
        print("_____________________________________________________________________________________________________")

    
    