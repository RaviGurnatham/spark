#from pyspark import sparkConf,sparkContext
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import NGram, CountVectorizer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession


def build_ngrams(inputCol="string", n=2):
    ngrams = [ NGram(n=i, inputCol="string", outputCol="{0}_grams".format(i))for i in range(1, n + 1)]
    vectorizers = [CountVectorizer(inputCol="{0}_grams".format(i),outputCol="{0}_counts".format(i))for i in range(1, n + 1)]
    assembler = [VectorAssembler(inputCols=["{0}_counts".format(i) for i in range(1, n + 1)],outputCol="features")]
    return Pipeline(stages=ngrams + vectorizers + assembler)






conf = SparkConf()
sc = SparkContext(conf = conf)
spark = SparkSession(sc)
rdd=sc.wholeTextFiles('/user/rey3012/train/bytes/*') #reading files
rdd = rdd.map(lambda x: x[1].split(' ')).map(lambda x: [i for i in x if(len(i)==2 and i!='??')])
rdd=rdd.map(lambda x: [x])
df=rdd.toDF(['string'])
cv=CountVectorizer(inputCol='string' , outputCol='feaures')
result=build_ngrams().fit(df).transform(df)
result_csv=result.toPandas()
result_csv.to_csv('complete_bigrams.csv')














import random

BOARD_SIZE = 3
goal_state = [1,2,3,4,5,6,7,8," "]
state1 = [5,2,7," ",4,8,3,6,1]
state2 = [1,2,3,4,5,6,7," ",8]
state3 = [1,2," ",5,6,3,4,7,8]
def show_state(state):
    for i in range(len(state)):
        print(state[i], end=" ")
        if i % 3 == 2:
            print()
    print()

def get_possible_actions(state):
    # return a list of actions. subset of ["Left","Right","Up","Down"]
    # get the position of blank.
    # First index of the black. Next row, col of the blank
    actions = []
    blank_index = state.index(" ")
    r = blank_index // 3
    c = blank_index % 3
    if r > 0:
        actions.append("Up")
    if r < 2:
        actions.append("Down")
    if c > 0:
        actions.append("Left")
    if c < 2:
        actions.append("Right")
    return actions

def update_state(state,action):
    # change the given state after taking the action
    blank_index = state.index(" ")
    if action == "Left":
        switch_index = blank_index - 1
    elif action == "Right":
        switch_index = blank_index + 1
    elif action == "Down":
        switch_index = blank_index + 3
    elif action == "Up":
        switch_index = blank_index - 3
    state[blank_index], state[switch_index] = state[switch_index], state[blank_index]

def random_shuffle(state,move_cnt):
    for i in range(move_cnt):
        action = random.choice(get_possible_actions(state))
        update_state(state,action)
    
def random_search(state):
    # change the given state until you arrive at the goal state
    move_count = 0
    while state != goal_state:
        random_action = random.choice(get_possible_actions(state))
        update_state(state,random_action)
        move_count += 1
        #print(random_action)
    show_state(state)
    print("Finished in {} steps.".format(move_count))

def expand(state):
    # return children states from the given state
    # For example
    # expand([1,2,3,4,5,6," ",7,8]) # possibles actions are ["Up", "Right"]
    #   returns [ [1,2,3," ",5,6,4,7,8], [1,2,3,4,5,6,7," ",8] ]
    successors = []
    for action in get_possible_actions(state):
        new_state = state[:]
        update_state(new_state,action)
        successors.append(new_state)
    return successors

def show_solution(node):
    path = [node[1]]
    while node[2]:
        node = node[2]
        path.append(node[1])
    path.reverse()
    print("The solutions is...")
    if len(path) < 10:
        for s in path:
            show_state(s)
    print("Finished in {} steps".format(len(path)-1))
        
def show_solution2(node):   # for AStar
    path = [node[3]]
    while node[-1]:
        node = node[-1]
        path.append(node[3])
    path.reverse()
    print("The solutions is...")
    if len(path) < 10:
        for s in path:
            show_state(s)
    print("Finished in {} steps".format(len(path)-1))
        
def BFS(initial_state): # Breadth-First Search
    # node is (cost, state, parent)
    visited_states = set()  #visited states till now
    root_node = (0,initial_state,None)  
    frontier = [root_node]  #no.of available ststes to expand
    loop_cnt = 0
    num_generated_nodes = 0 
    while frontier != []:  #run till fronties is empty
        loop_cnt += 1
        node = frontier.pop(0) #first elemnt in the frontier (state) --> tuple  (cost, state, parent)
        #print(type(node))
        if node[1] == goal_state: #if node[1] isgoal state then no need of further exploration.
            show_solution(node)                        
            print(loop_cnt, num_generated_nodes, len(visited_states))
            return
        # expand the state. add the successor to the frontier
        
        successors = expand(node[1]) # thsi functin will expand all possible ways
        for succ in successors: # taking child from all possible ways from the parent.
            if tuple(succ) not in visited_states:  #
                visited_states.add(tuple(succ))
                new_node = (node[0]+1, succ, node)
                frontier.append(new_node)
            num_generated_nodes += 1

def DFS(initial_state): # Depth-First Search
    # node is (cost, state, parent)
    visited_states = set()
    root_node = (0,initial_state,None)
    frontier = [root_node]
    loop_cnt = 0
    num_generated_nodes = 0 
    while frontier != []:
        loop_cnt += 1
        node = frontier.pop(0)
        if node[1] == goal_state:
            show_solution(node)
            print(loop_cnt, num_generated_nodes, len(visited_states))
            return True
        # expand the state. add the successor to the frontier
        successors = expand(node[1])
        for succ in successors:
            if tuple(succ) not in visited_states:
                visited_states.add(tuple(succ))
                new_node = (node[0]+1, succ, node)
                frontier.insert(0,new_node)
            num_generated_nodes += 1

def DFS_limited(initial_state, max_depth): # Depth-First Search


       # node is (cost, state, parent)
    visited_states = []
    visited_cost={}
    root_node = (0,initial_state,None)
    frontier = [root_node]
    loop_cnt = 0
    num_generated_nodes = 0 
    while frontier != [] and num_generated_nodes<max_depth:
        loop_cnt += 1
        node = frontier.pop(0)
        if node[1] == goal_state:
            show_solution(node)
            print(loop_cnt, num_generated_nodes, len(visited_states))
            return True
        successors = expand(node[1])
        for succ in successors:
            if tuple(succ) in visited_states :
                previous_cost=visited_cost[tuple(succ)]
            #    print('previous cost is ', previous_cost, 'current_cost is',node[0])
                visited_cost[tuple(succ)]=min(previous_cost,node[0])
               # show_state(tuple(succ))  
            else:
                visited_states.append(tuple(succ))
                visited_cost[tuple(succ)]=node[0]+1
                new_node = (node[0]+1, succ, node)
                frontier.insert(0,new_node)
               # show_state(tuple(succ))

            num_generated_nodes += 1
    else:
        return False
    
    # similar to DFS but does not go deeper than max_depth

    # visited_states need to keep cost for each state.  
    # eg) visited_states = {} # set of (k,v) pairs. k is tuple(board), v is cost 
    # if a generated states is in the visited_states, but if it has smaller cost
    # then still add a node of the board while updating the cost in visited_states.
	
    # It is possible to fail to find solution. Return False when it fails, True when succeed.
    pass # replace this line with your code

def DFS_iterative_deepening(initial_state):
    # Call DFS_limited multiple times until it succeed.
    for max_depth in range(1,1002,100):
        print('Evaluating with depth ', max_depth)
        result=DFS_limited(initial_state, max_depth)
        if result:
            return result
    else:
        return False
    # Begin with max_depth = 1 and increase it by one 
    # Refer to the pseudo code in page 89 of the textbook.


def heuristic(state):
    count = 0;
    for i in range(len(state)):
##       if state[i] != " " and i!=state[i]-1:
        if state[i] != goal_state[i]:
           count += 1            
    return count

import time
from heapq import *
def AStar(initial_state):
    # node is (g+h,h,time.perf_counter(),state, parent)
    visited_states = {}
    h = heuristic(initial_state)
    root_node = (0+h,h,time.perf_counter(),initial_state,None)
    frontier = [root_node]
    loop_cnt = 0
    num_generated_nodes = 0 
    while frontier != []:
        loop_cnt += 1
        node = heappop(frontier)
        if node[3] == goal_state:
            show_solution2(node)
            print(loop_cnt, num_generated_nodes, len(visited_states))
            return
        # expand the state. add the successor to the frontier
        successors = expand(node[3])
        for succ in successors:
            h = heuristic(succ)
            node_g = node[0]-node[1]
            g = node_g + 1
            if tuple(succ) not in visited_states or g+h < visited_states[tuple(succ)]:
                visited_states[tuple(succ)] = g + h
                new_node = (g+h,h, time.perf_counter(), succ, node)
##                frontier.append(new_node)
##                heapify(frontier)
                heappush(frontier,new_node)
            num_generated_nodes += 1

#print(heuristic([1,2,3,4,5," ",6,7,8]))
   
#print(expand([1,2,3,4," ", 5,6,7,8]))
#state1 = goal_state[:]
#random_shuffle(state1,100)
#show_state(state1)
#random_search(state3)

#DFS(state3)
#AStar(state3)
show_state(state3)
print(get_possible_actions(state3))
#update_state(state3,"Down")
#show_state(state3)
#random_search(state3)
#result=DFS_limited(state3,20)
result=DFS_iterative_deepening(state3)
print(result)
#print(result)
#DFS(state3)
##childs=expand(state3)
##for child in childs:
##    show_state(child)
              
#show_state(state1)



























