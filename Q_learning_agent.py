import random
import pickle
import numpy as np

class Agent:

    # decided rewards by the participants
    EARN_TREASURE               =       30           # (unused) treasure removed from the game
    DESTROY_SOFTBLOCK           =       3       
    DESTROY_ORE                 =       7     
    DO_DAMAGE                   =       500
    TAKE_DAMAGE                 =       -300       
    EARN_AMMO                   =       3
    WASTED_MOVE                 =      -5 

    #   "gamma = discount factor. High gamma value means focus on future rewards "
    gamma = 0.99
    #   "learning rate, High learning rate means faster learning"                     
    LR = 0.001

    def __init__(self):
        '''
        Place any initialization code for your agent here (if any)
        '''
        self.return_sum = 0                     # this stores sum total reward in every iteration
        self.N = self.initializeN()             # sets the episode number

        # the following variables are declared in __init__ because they need to survive in the upcoming tick for training purpose
        self.old_state = None                   #  old state (relevant for learning)
        self.old_action = None                  #  old action (relevant for learning )
        self.Q = self.initializeQ()             #  Q Table
        self.epsilon = self.initializeE()       #  epsilon
        pass

#-------------------------------------------------------------------------------------------
    # initialization functions to store data in case participant wants to stop training now and resume later
    def initializeN(self):
        try:
            with open('Q_Learning_G_Number', 'rb') as f:
                n = pickle.load(f)
                #print("model loaded")
        except:
                n = 0
                with open("Q_Learning_G_Number","wb") as f:
                    pickle.dump(n, f)
        return n

    def initializeQ(self):
        try:
            with open('Q_Learning_Q_TABLE', 'rb') as f:
                q = pickle.load(f)
                #print("model loaded")
        except:
                q = dict()
                with open("Q_Learning_Q_TABLE","wb") as f:
                    pickle.dump(q, f)
        return q
    
    def initializeE(self):
        try:
            with open('Q_Learning_epsilon', 'rb') as f:
                q = pickle.load(f)
                #print("model loaded")
        except:
            q = 1
            with open("Q_Learning_epsilon","wb") as f:
                    pickle.dump(q, f)
        return q
#-----------------------------------------------------------------------------------------------------------
    def learn(self, state, state2, reward, action, action2):
        action = action.index(1)     # convert old action from list([0,0,1,0,0,0]) to index number
        action2 = action2.index(1)   # convert new action from list([0,0,1,0,0,0]) to index number

        # if state already in Q table, then it means that particlar state has been explored, otherwise, new entry is added to the table
        if(state not in self.Q):                     
            self.Q[state] = np.random.uniform(0,1,6)
        if(state2 not in self.Q):
            self.Q[state2] = np.random.uniform(0,1,6)
        # calculate predict and target
        predict = self.Q[state][action].item()      # valuee for particular state and action given by the state
        target = reward + self.gamma * np.argmax(self.Q[state2]).item()     # optimal value
        #update the q value for the particular state and action 
        self.Q[state][action] = self.Q[state][action].item() + self.LR * (target - predict) #updating the self.q_value


#--------------------------------------------------------------------------------------------------------------
    #helper function for training_state to identify whether the agent is safe or not
    #   detects possible danger areas marked by bomb explosions
    def is_in_range(self, location, bombs, game_state):
        bombs_in_range = []
        if(game_state.is_in_bounds(location)==False):
            return True
        for bomb in bombs:
            distance = self.manhattan_distance(location, bomb)
            if(distance<=10):
                return True 
        return False
#--------------------------------------------------------------------------------------------------------------------
    #epsilon greedy method to determine which action to choose. 
    #   a random number is generated between 0 and 1. 
    #   if it's smaller than epsilon, then a random move is implemented
    #   else action is taken from the q table
    def get_action(self, state):
        final_move =[0,0,0,0,0,0]
        if np.random.uniform(0, 1) < self.epsilon:
            move = random.randint(0, 5)
            final_move[move] = 1
        else:
            if(state not in self.Q):
                move = random.randint(0, 5)
                final_move[move] = 1
                return final_move
            move = np.argmax(self.Q[state])
            final_move[move] = 1
        return final_move

#----------------------------------------------------------------------------------------------------------
    def next_move(self, game_state, player_state):
        '''
        This method is called each time your Agent is required to choose an action
        '''

        # if game tick is 0, there is no old_state or old_action to train the agent, therefore, we just pass this state without training
        # we set value for old_action and old_state from the tick _number state
        if(game_state.tick_number==0):
            self.old_state = self.calculate_training_state(game_state, player_state)
            ac = self.get_action(self.old_state)
            self.old_action  = ac
            ac = ['','u','d','l','r','p'][ac.index(1)]
            return ac

        # initialize values for training state
        reward = self.calculate_reward_for_move(game_state)     #  reward obtained from old_action (not sum reward)
        new_state = self.calculate_training_state(game_state, player_state) # new state because of old_action
        new_action = self.get_action(new_state)                 # action to be implemented in current state

        # just extra reward adjustment for telling the agent not to use p when there are no bombs
        if(player_state.ammo==0 and new_action=="p"):           # check if player still has ammo
            self.reward = self.reward - 10

        self.return_sum+=reward             # update return_sum for calculating cumulative reward
        #print([self.old_state, new_state, reward, self.old_action, new_action])
        self.learn(self.old_state, new_state, reward, self.old_action, new_action) #learn method to learn from the state
        if(game_state.is_over):
            self.N+=1
            self.epsilon-=0.002

            # store the information from the agent
            with open("Q_Learning_Q_TABLE","wb") as Q_table:
                pickle.dump(self.Q, Q_table)

            with open("Q_Learning_G_Number", "wb") as f:
                pickle.dump(self.N, f)
            
            with open("Q_Learning_epsilon", "wb") as f:
                pickle.dump(self.epsilon, f)

            print("Game Number : ", self.N)
            print("Reward Earned : ", self.return_sum)
            self.return_sum = 0
        self.old_action = new_action  
        self.old_state = new_state
        #print(self.Q)
        new_action = ['','u','d','l','r','p'][new_action.index(1)]

        
        return new_action


    def calculate_training_state(self, game_state, player_state):
        training_state = []
        x = player_state.location[0]
        y = player_state.location[1]
        fl = (x-2, y)
        ft = (x, y+2)
        fr = (x+2, y)
        fb = (x, y-2)
        nl = (x-1,y)
        nt = (x,y+1)
        nr = (x+1,y)
        nb = (x,y-1)
        own = (x,y)
        state_space = [fl,ft,fr,fb,nl,nt,nr,nb,own]
        in_range = False
        for pt in state_space:
            p_ = game_state.entity_at(pt)
            if(game_state.is_in_bounds(pt) == False):
                training_state.append("0")
            elif(p_==None): # there is no object there
                training_state.append("1")
            elif(p_=="sb" or "ob"):
                training_state.append("2")
            elif(p_=="a"):
                training_state.append("3")
            elif(p_=="b"):
                training_state.append("4")
            elif(p_=="1" or p_=="0"):
                training_state.append("5")
            if(self.is_in_range(pt, game_state.bombs, game_state)==True):
                in_range = True
        
        x_diff = game_state.opponents(player_state.id)[0][0]-player_state.location[0] #give oppponent locations
        y_diff = game_state.opponents(player_state.id)[0][1]-player_state.location[1] #give oppponent locations
        training_state.append(str(int((abs(x_diff)+abs(y_diff))*0.9-1)))
        if(in_range ==True):
            training_state.append("1")
        else:
            training_state.append("0")

        if(player_state.ammo==0):
            training_state.append("0")
        else:
            training_state.append("1")
        
        training_state = ''.join(training_state)
        return training_state

    def calculate_reward_for_move(self, game_state):
        reward = 0
        if(game_state._occurred_event[0]==1):
            #playered earned a treasure 
            reward+=self.EARN_TREASURE
            
        if(game_state._occurred_event[1]>0):
            #playered broke a wooden block
            reward+=(self.DESTROY_SOFTBLOCK*game_state._occurred_event[1])
            
        if(game_state._occurred_event[2]>0):
            #played broken an ore
            reward+=(self.DESTROY_ORE*game_state._occurred_event[2])
            
        if(game_state._occurred_event[3]==1):
            reward+=self.DO_DAMAGE
            #player did damage
            
        if(game_state._occurred_event[4]==1):
            reward+=self.TAKE_DAMAGE
            
        
        if(game_state._occurred_event[5]==1):
            #player has earned some ammo
            reward+=self.EARN_AMMO
            

        if(game_state._occurred_event[6]==1):
            #wasted move has been made
            reward+=self.WASTED_MOVE
            

        return reward

        """
        1. time passed since last damage sufferred.
        2. time passed since last hit made.
        6. area of map explored
        8. 
        """


    def manhattan_distance(self, start, end):
        distance = abs(start[0]-end[0] + abs(start[1]-end[1]))
        return distance       